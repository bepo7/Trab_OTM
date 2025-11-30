import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import math

# --- IMPORTAÇÕES DO SEU PROJETO ---
import preparar_dados
import otimizar
import config

def safe_float(val):
    try:
        val = float(val)
        if math.isnan(val) or math.isinf(val): return 0.0
        return val
    except: return 0.0

def limpar_string(s):
    """Remove espaços e padroniza para garantir o match"""
    return str(s).strip().upper()

# ==============================================================================
# SOLVER GUROBI COM PÓS-PROCESSAMENTO DE SEGURANÇA
# ==============================================================================
def resolver_com_gurobi_setores(inputs, lambda_risk, risco_max_usuario, 
                                warm_start_pesos, setores_proibidos,
                                teto_maximo_ativo=0.30,
                                teto_maximo_setor=1.0):
    
    # 1. Extração de Dados
    retornos = inputs['retornos_medios'].values
    cov_matrix = inputs['matriz_cov'].values
    vals_pvp = inputs['vetor_pvp'].values
    vals_cvar = inputs['vetor_cvar'].values
    nomes_ativos = inputs['nomes_dos_ativos']
    n_ativos = len(retornos)
    
    volume_medio = inputs['volume_medio']
    valor_investido = inputs['valor_total_investido'] or 1.0
    MIN_PESO_ATIVO = 0.005

    print(f"\n[GUROBI] Iniciando... Teto Ativo: {teto_maximo_ativo:.1%} | Teto Setor: {teto_maximo_setor:.1%}")

    # --- MAPEAMENTO DE SETORES (NORMALIZADO) ---
    mapa_setores = config.obter_mapa_setores_ativos()
    indices_por_setor = {}
    
    # Cria um mapa auxiliar limpo: { "PETR4.SA": "Energia", ... }
    ativo_para_setor = {}

    for setor, lista_ativos in mapa_setores.items():
        if setor not in indices_por_setor: indices_por_setor[setor] = []
        for ativo in lista_ativos:
            ativo_limpo = limpar_string(ativo)
            ativo_para_setor[ativo_limpo] = setor

    # Preenche os índices com base na ordem de nomes_ativos
    ativos_sem_setor = 0
    for i, ticker in enumerate(nomes_ativos):
        ticker_limpo = limpar_string(ticker)
        if ticker_limpo in ativo_para_setor:
            setor = ativo_para_setor[ticker_limpo]
            indices_por_setor[setor].append(i)
        else:
            ativos_sem_setor += 1
            # DEBUG: Se for FII e não achou, avisa
            if "11" in ticker_limpo: 
                print(f"   [AVISO] Ativo {ticker} não foi encontrado no mapa de setores!")

    # Mapeamento de Proibidos
    ativos_proibidos_set = set()
    if setores_proibidos:
        for setor in setores_proibidos:
            if setor in mapa_setores:
                ativos_proibidos_set.update([limpar_string(a) for a in mapa_setores[setor]])

    # --- MODELO ---
    model = gp.Model("Portfolio_MultiObj")
    model.setParam('OutputFlag', 0)
    
    pesos = []
    binarias = []
    
    for i, ticker in enumerate(nomes_ativos):
        ticker_limpo = limpar_string(ticker)
        
        # A. Liquidez
        vol = safe_float(volume_medio.get(ticker, 0.0))
        max_liq = safe_float((0.1 * vol) / valor_investido)
        
        # B. Limite Superior
        # O limite é o MENOR entre: Teto Ativo, Teto Setor, Liquidez
        limite_ub = min(teto_maximo_ativo, teto_maximo_setor, max_liq)
        
        if ticker_limpo in ativos_proibidos_set:
            limite_ub = 0.0
            
        # C. Variável de Peso (Contínua)
        if limite_ub < MIN_PESO_ATIVO:
             p = model.addVar(lb=0.0, ub=0.0, vtype=GRB.CONTINUOUS)
             z = model.addVar(lb=0.0, ub=0.0, vtype=GRB.BINARY) # Força 0
        else:
            p = model.addVar(lb=0.0, ub=limite_ub, vtype=GRB.CONTINUOUS) # LB é 0 pois é controlado pelo Z
            z = model.addVar(vtype=GRB.BINARY)

        # D. Warm Start
        if warm_start_pesos is not None:
             val = warm_start_pesos[i]
             if val > MIN_PESO_ATIVO:
                 p.Start = min(val, limite_ub)
                 z.Start = 1.0
             else:
                 p.Start = 0.0
                 z.Start = 0.0

        pesos.append(p)
        binarias.append(z)
        
    model.update()

    # Objetivos e Restrições Globais
    expr_ret = gp.quicksum(pesos[i] * retornos[i] for i in range(n_ativos))
    expr_var = gp.quicksum(pesos[i] * cov_matrix[i, j] * pesos[j] for i in range(n_ativos) for j in range(n_ativos))
    expr_pvp = gp.quicksum(pesos[i] * vals_pvp[i] for i in range(n_ativos))
    expr_cvar = gp.quicksum(pesos[i] * vals_cvar[i] for i in range(n_ativos))
    
    # Penalidade por Caixa (1.0 - soma_pesos)
    expr_soma_pesos = gp.quicksum(pesos)
    expr_penalidade_caixa = config.PESO_PENALIZACAO_CAIXA * (1.0 - expr_soma_pesos)

    # Objetivo: Mean-Variance com penalidades (sem Sharpe para manter convexidade)
    obj = (lambda_risk * expr_var) - expr_ret + (config.PESO_PVP * expr_pvp) + (config.PESO_CVAR * expr_cvar) + expr_penalidade_caixa
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Restrição de Orçamento: Soma(w) <= 1.0 (Permite caixa)
    model.addConstr(gp.quicksum(pesos) <= 1.0, "orcamento")
    model.addConstr(expr_var <= risco_max_usuario ** 2, "Risco")
    
    # Restrições de Setor
    if teto_maximo_setor < 0.999:
        for setor, idxs in indices_por_setor.items():
            if idxs:
                model.addConstr(gp.quicksum(pesos[i] for i in idxs) <= teto_maximo_setor, f"TetoSetor_{setor}")

    # Restrições de Cardinalidade / Peso Mínimo (Big-M formulation)
    # Se z[i] = 0 => w[i] = 0
    # Se z[i] = 1 => 0.005 <= w[i] <= teto_efetivo
    LIMITE_MINIMO = 0.005
    
    for i in range(n_ativos):
        ticker_limpo = limpar_string(nomes_ativos[i])
        
        # A. Liquidez
        vol = safe_float(volume_medio.get(nomes_ativos[i], 0.0))
        max_liq = safe_float((0.1 * vol) / valor_investido)
        
        # B. Limite Superior
        # O limite é o MENOR entre: Teto Ativo, Teto Setor, Liquidez
        limite_ub_i = min(teto_maximo_ativo, teto_maximo_setor, max_liq)
        
        if ticker_limpo in ativos_proibidos_set:
            limite_ub_i = 0.0
            
        # Restrição Superior (Big-M): w[i] <= UB * z[i]
        model.addConstr(pesos[i] <= limite_ub_i * binarias[i], f"upper_bound_bin_{i}")
        
        # Restrição Inferior (Threshold): w[i] >= 0.005 * z[i]
        # Only apply if the upper bound allows for a non-zero weight
        if limite_ub_i >= LIMITE_MINIMO:
            model.addConstr(pesos[i] >= LIMITE_MINIMO * binarias[i], f"lower_bound_bin_{i}")
        else: # If UB is too small, force w[i] to 0 and z[i] to 0
            model.addConstr(pesos[i] == 0, f"force_zero_w_{i}")
            model.addConstr(binarias[i] == 0, f"force_zero_z_{i}")

    model.optimize()
    
    if model.Status == GRB.OPTIMAL:
        w_otimo = np.array([pesos[i].X for i in range(n_ativos)])
        
        # --- PÓS-PROCESSAMENTO DE SEGURANÇA (O CORRETOR MANUAL) ---
        # Se o Gurobi falhar ou soltar algo errado, cortamos na marra.
        
        # 1. Corta ativos individuais que estourarem o teto (Ativo ou Setor)
        # O limite efetivo é o menor entre Teto Ativo e Teto Setor
        teto_efetivo = min(teto_maximo_ativo, teto_maximo_setor)
        
        mask_estouro_individual = w_otimo > (teto_efetivo + 0.001)
        if np.any(mask_estouro_individual):
            print(f"🚨 [CORREÇÃO] Forçando corte em ativos que excederam {teto_efetivo:.1%}")
            w_otimo[mask_estouro_individual] = teto_efetivo

            # 2. Corta a soma do setor se estourar
            for setor, idxs in indices_por_setor.items():
                soma_setor = w_otimo[idxs].sum()
                if soma_setor > (teto_maximo_setor + 0.001):
                    print(f"🚨 [CORREÇÃO] Setor {setor} estourou ({soma_setor:.1%}). Normalizando...")
                    fator = teto_maximo_setor / soma_setor
                    w_otimo[idxs] *= fator
        
        # Recalcula métricas com os pesos corrigidos
        ret_final = np.dot(w_otimo, retornos)
        var_final = np.dot(w_otimo, np.dot(cov_matrix, w_otimo))
        pvp_final = np.dot(w_otimo, vals_pvp)
        cvar_final = np.dot(w_otimo, vals_cvar)
        sharpe_final = ret_final / (np.sqrt(var_final) + 1e-9)
        
        return {
            'pesos': w_otimo,
            'obj': model.ObjVal,
            'retorno': ret_final,
            'risco': np.sqrt(var_final),
            'pvp_final': pvp_final,
            'cvar_final': cvar_final,
            'sharpe': sharpe_final
        }
    else:
        print(f"[GUROBI] Falha. Status: {model.Status}")
        return None