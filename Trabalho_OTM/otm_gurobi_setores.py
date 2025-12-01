import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import math

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
    return str(s).strip().upper()

def resolver_com_gurobi_setores(inputs, lambda_risk, risco_max_usuario, 
                                warm_start_pesos, setores_proibidos,
                                teto_maximo_ativo=0.30,
                                teto_maximo_setor=1.0,
                                max_ativos_carteira=15, # <--- PARÂMETRO NOVO
                                max_ativos_setor=4):    # <--- PARÂMETRO NOVO
    
    # --- PARÂMETROS ---
    MIN_PESO_SE_COMPRAR = 0.005
    # ------------------

    retornos = inputs['retornos_medios'].values
    cov_matrix = inputs['matriz_cov'].values
    vals_pvp = inputs['vetor_pvp'].values
    vals_cvar = inputs['vetor_cvar'].values
    nomes_ativos = inputs['nomes_dos_ativos']
    n_ativos = len(retornos)
    
    volume_medio = inputs['volume_medio']
    valor_investido = inputs['valor_total_investido'] or 1.0
    
    if 'ultimos_precos' in inputs:
        precos_atuais = inputs['ultimos_precos']
    else:
        print("⚠️ AVISO: Preços atuais não encontrados. Usando fallback de R$ 10,00.")
        precos_atuais = pd.Series([10.0] * n_ativos, index=nomes_ativos)

    print(f"\n[GUROBI] Iniciando... Max Global: {max_ativos_carteira} | Max/Setor: {max_ativos_setor}")

    # Mapeamento
    mapa_setores = config.obter_mapa_setores_ativos()
    indices_por_setor = {}
    ativo_para_setor = {}

    for setor, lista_ativos in mapa_setores.items():
        if setor not in indices_por_setor: indices_por_setor[setor] = []
        for ativo in lista_ativos:
            ativo_limpo = limpar_string(ativo)
            ativo_para_setor[ativo_limpo] = setor

    for i, ticker in enumerate(nomes_ativos):
        ticker_limpo = limpar_string(ticker)
        if ticker_limpo in ativo_para_setor:
            setor = ativo_para_setor[ticker_limpo]
            indices_por_setor[setor].append(i)

    ativos_proibidos_set = set()
    if setores_proibidos:
        for setor in setores_proibidos:
            if setor in mapa_setores:
                ativos_proibidos_set.update([limpar_string(a) for a in mapa_setores[setor]])

    # --- MODELO ---
    model = gp.Model("Portfolio_Dinamico")
    model.setParam('OutputFlag', 0) 
    
    pesos = []
    vars_lotes = []
    vars_binarias = [] 

    for i, ticker in enumerate(nomes_ativos):
        ticker_limpo = limpar_string(ticker)
        
        preco_unitario = safe_float(precos_atuais.get(ticker, 0.0))
        custo_acao = preco_unitario
        vol = safe_float(volume_medio.get(ticker, 0.0))
        
        teto_financeiro_ativo = min(valor_investido * teto_maximo_ativo, 0.1 * vol)
        if ticker_limpo in ativos_proibidos_set: teto_financeiro_ativo = 0.0
            
        if custo_acao <= 0.01: 
            max_unidades = 0
        else:
            max_unidades = int(teto_financeiro_ativo / custo_acao)
        
        min_financeiro_ativo = valor_investido * MIN_PESO_SE_COMPRAR
        min_cotas = math.ceil(min_financeiro_ativo / (custo_acao + 0.0001))

        if min_cotas > max_unidades:
            max_unidades = 0
            min_cotas = 0

        z = model.addVar(vtype=GRB.BINARY, name=f"bin_{ticker}")
        vars_binarias.append(z)
        
        n_cotas = model.addVar(lb=0, ub=max_unidades, vtype=GRB.INTEGER, name=f"qtd_{ticker}")
        vars_lotes.append(n_cotas)

        if warm_start_pesos is not None:
            peso_ga = warm_start_pesos[i]
            
            if peso_ga > 1e-6: # Só tenta Warm Start se o GA alocou algo relevante
                # 1. Valor financeiro sugerido pelo GA
                valor_sugerido = peso_ga * valor_investido
                
                # 2. Converte para Quantidade de Ações (Arredonda para baixo)
                if custo_acao > 0.01:
                    qtd_sugerida = int(valor_sugerido / custo_acao)
                    
                    # 3. Respeita os limites do Gurobi (caso a estimativa do GA tenha excedido)
                    if qtd_sugerida > max_unidades: qtd_sugerida = max_unidades
                    
                    # 4. Define o Ponto de Partida (.Start) para as cotas inteiras
                    n_cotas.Start = qtd_sugerida
                    
                    # 5. Liga a binária se a quantidade sugerida for válida (acima do mínimo)
                    if qtd_sugerida >= min_cotas and qtd_sugerida > 0:
                        z.Start = 1.0
                    else:
                        z.Start = 0.0
                else:
                    z.Start = 0.0
            else:
                z.Start = 0.0 # Garante que o Warm Start não force a compra se o peso for zero

        model.addConstr(n_cotas <= max_unidades * z, f"link_max_{ticker}")
        model.addConstr(n_cotas >= min_cotas * z, f"link_min_{ticker}")

        if valor_investido > 0:
            peso_expr = (n_cotas * custo_acao) / valor_investido
        else:
            peso_expr = 0.0
        pesos.append(peso_expr)
        
    model.update()

    # --- OBJETIVOS E RESTRIÇÕES GLOBAIS ---
    
    expr_ret = gp.quicksum(pesos[i] * retornos[i] for i in range(n_ativos))
    expr_var = gp.quicksum(pesos[i] * cov_matrix[i, j] * pesos[j] for i in range(n_ativos) for j in range(n_ativos))
    expr_pvp = gp.quicksum(pesos[i] * vals_pvp[i] for i in range(n_ativos))
    expr_cvar = gp.quicksum(pesos[i] * vals_cvar[i] for i in range(n_ativos))
    expr_soma_pesos = gp.quicksum(pesos)
    
    obj = (lambda_risk * expr_var) - expr_ret + (config.PESO_PVP * expr_pvp) + (config.PESO_CVAR * expr_cvar) + (config.PESO_PENALIZACAO_CAIXA * (1.0 - expr_soma_pesos))
    model.setObjective(obj, GRB.MINIMIZE)
    
    model.addConstr(expr_soma_pesos <= 1.0, "orcamento")
    model.addConstr(expr_var <= risco_max_usuario ** 2, "Risco")
    
    if teto_maximo_setor < 0.999:
        for setor, idxs in indices_por_setor.items():
            if idxs:
                model.addConstr(gp.quicksum(pesos[i] for i in idxs) <= teto_maximo_setor, f"TetoFin_{setor}")

    # --- RESTRIÇÕES DE CARDINALIDADE (DINÂMICAS) ---

    # 1. Global
    model.addConstr(gp.quicksum(vars_binarias) <= max_ativos_carteira, "Card_Global")

    # 2. Por Setor
    for setor, idxs in indices_por_setor.items():
        if idxs:
            model.addConstr(gp.quicksum(vars_binarias[i] for i in idxs) <= max_ativos_setor, f"Card_Setor_{setor}")

    model.optimize()
    
    if model.Status == GRB.OPTIMAL:
        w_otimo = np.zeros(n_ativos)
        lotes_otimos = np.zeros(n_ativos)
        
        for i in range(n_ativos):
            qtd = vars_lotes[i].X
            preco = safe_float(precos_atuais.get(nomes_ativos[i], 0.0))
            if valor_investido > 0:
                w_otimo[i] = (qtd * preco) / valor_investido
            lotes_otimos[i] = qtd
        
        ret_final = np.dot(w_otimo, retornos)
        var_final = np.dot(w_otimo, np.dot(cov_matrix, w_otimo))
        pvp_final = np.dot(w_otimo, vals_pvp)
        cvar_final = np.dot(w_otimo, vals_cvar)
        
        investido_real = w_otimo.sum() * valor_investido

        if investido_real > valor_investido:
            investido_real = valor_investido
        
        sobra = valor_investido - investido_real

        qtd_ativos_selecionados = int(sum(vars_binarias[i].X for i in range(n_ativos)))
        print(f"   > [SUCESSO] Inv: R$ {investido_real:.2f} | Sobra: R$ {sobra:.2f} | Ativos: {qtd_ativos_selecionados}")
        
        return {
            'pesos': w_otimo,
            'lotes': lotes_otimos,
            'obj': model.ObjVal,
            'retorno': ret_final,
            'risco': np.sqrt(var_final),
            'pvp_final': pvp_final,
            'cvar_final': cvar_final
        }
    else:
        print(f"[GUROBI] Falha. Status: {model.Status}")
        return None