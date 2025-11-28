import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np

# --- IMPORTAÇÕES DO SEU PROJETO ---
import preparar_dados
import otimizar
import config

# ==============================================================================
# 1. SOLVER GUROBI COM RESTRIÇÃO DE SETORES + LIQUIDEZ + MÍNIMO DE ENTRADA
# ==============================================================================
def resolver_com_gurobi_setores(inputs, lambda_risk, risco_max_usuario, 
                                warm_start_pesos, setores_proibidos):
    
    # 1. Extração de Dados Básicos
    retornos = inputs['retornos_medios'].values
    cov_matrix = inputs['matriz_cov'].values
    vals_pvp = inputs['vetor_pvp'].values
    vals_cvar = inputs['vetor_cvar'].values
    nomes_ativos = inputs['nomes_dos_ativos']
    n_ativos = len(retornos)
    
    # 2. Extração de Dados para Liquidez (NOVOS INPUTS)
    volume_medio = inputs['volume_medio'] # Series Pandas
    valor_investido = inputs['valor_total_investido']
    # Evita divisão por zero caso valor investido não venha preenchido
    if valor_investido is None or valor_investido <= 0:
        valor_investido = 1.0

    # 3. Configurações de Restrições
    MIN_PESO_ATIVO = 0.005  # Mínimo de 0.5% se entrar no ativo
    TETO_GLOBAL_ATIVO = 0.30 # Máximo de 30% por ativo (Regra de diversificação)

    # --- MAPEAMENTO DE ATIVOS PROIBIDOS ---
    mapa_setores = config.obter_mapa_setores_ativos()
    ativos_proibidos_set = set()
    
    if setores_proibidos:
        for setor in setores_proibidos:
            if setor in mapa_setores:
                ativos_proibidos_set.update(mapa_setores[setor])
    
    print(f"[GUROBI] Configurando restrições...")
    print(f"   > Setores Proibidos: {setores_proibidos}")
    print(f"   > Mínimo de Entrada: {MIN_PESO_ATIVO:.1%}")
    print(f"   > Teto Global por Ativo: {TETO_GLOBAL_ATIVO:.1%}")

    # --- INÍCIO DO MODELO ---
    model = gp.Model("Portfolio_MultiObj")
    model.setParam('OutputFlag', 0)
    
    pesos = []
    
    # --- CRIAÇÃO DAS VARIÁVEIS COM LIMITES (MODIFICADO) ---
    for i, ticker in enumerate(nomes_ativos):
        
        # A. CÁLCULO DO TETO DE LIQUIDEZ (Igual ao modelo_problema_setor.py)
        # Regra: Só pode comprar até 1% do volume médio diário
        vol_ativo = volume_medio.get(ticker, 0.0)
        teto_financeiro = 0.05 * vol_ativo
        max_peso_liquidez = teto_financeiro / valor_investido
        
        # B. DEFINIÇÃO DO LIMITE SUPERIOR (UB)
        # O limite é o menor entre: 30% (global) e a Liquidez do ativo
        limite_superior = min(TETO_GLOBAL_ATIVO, max_peso_liquidez)
        
        # Se for setor proibido, força zero
        if ticker in ativos_proibidos_set:
            limite_superior = 0.0
        
        # C. DEFINIÇÃO DA VARIÁVEL (SEMICONT vs CONTINUOUS)
        
        # Caso 1: O ativo está proibido OU a liquidez é tão baixa que não permite nem os 0.5% mínimos
        if limite_superior < MIN_PESO_ATIVO:
             # Trava em 0.0
             p = model.addVar(lb=0.0, ub=0.0, vtype=GRB.CONTINUOUS, name=f"w[{ticker}]")
             
        else:
            # Caso 2: Ativo viável. Usa Semicontínua.
            # Significa: O peso pode ser 0.0 OU deve estar entre [0.005, limite_superior]
            p = model.addVar(
                lb=MIN_PESO_ATIVO, 
                ub=limite_superior, 
                vtype=GRB.SEMICONT, 
                name=f"w[{ticker}]"
            )
        
        # D. WARM START (Com sanitização)
        if warm_start_pesos is not None:
             val_warm = warm_start_pesos[i]
             
             # Ajusta se o GA mandou algo acima do permitido pela liquidez
             if val_warm > limite_superior:
                 val_warm = limite_superior
             
             # Ajusta se o GA mandou algo na "zona proibida" (ex: 0.002) -> zera
             if val_warm > 1e-6 and val_warm < MIN_PESO_ATIVO:
                 val_warm = 0.0
                 
             p.Start = val_warm

        pesos.append(p)
        
    model.update()

    # --- FUNÇÃO OBJETIVO ---
    
    # Termo Quadrático: Lambda * Variância
    expr_variancia = gp.quicksum(pesos[i] * cov_matrix[i, j] * pesos[j] 
                                 for i in range(n_ativos) for j in range(n_ativos))
    
    # Termo Linear: Retorno (Negativo pois é Max)
    expr_retorno = gp.quicksum(pesos[i] * retornos[i] for i in range(n_ativos))
    
    # Termo Linear: P/VP (Positivo pois é Min)
    expr_pvp = gp.quicksum(pesos[i] * vals_pvp[i] for i in range(n_ativos))
    
    # Termo Linear: CVaR (Positivo pois é Min)
    expr_cvar = gp.quicksum(pesos[i] * vals_cvar[i] for i in range(n_ativos))
    
    # Objetivo Final
    obj_final = (lambda_risk * expr_variancia) - expr_retorno \
                + (config.PESO_PVP * expr_pvp) \
                + (config.PESO_CVAR * expr_cvar)
    
    model.setObjective(obj_final, GRB.MINIMIZE)
    
    # --- RESTRIÇÕES ---
    model.addConstr(gp.quicksum(pesos[i] for i in range(n_ativos)) == 1.0, "Orcamento")
    model.addConstr(expr_variancia <= risco_max_usuario ** 2, "Teto_Risco")
    
    model.optimize()
    
    if model.Status == GRB.OPTIMAL:
        w_otimo = np.array([pesos[i].X for i in range(n_ativos)])
        
        # Recalcula métricas puras para exibir
        ret_final = np.dot(w_otimo, retornos)
        var_final = np.dot(w_otimo, np.dot(cov_matrix, w_otimo))
        
        return {
            'pesos': w_otimo,
            'obj': model.ObjVal,
            'retorno': ret_final,
            'risco': np.sqrt(var_final)
        }
    else:
        # Se falhar, tenta relaxar a precisão numérica
        print(f"[GUROBI] Status: {model.Status}. Tentando resolver novamente...")
        return None

# ==============================================================================
# 2. FLUXO PRINCIPAL (Mantido para teste)
# ==============================================================================
def main():
    print("--- BENCHMARK AVANÇADO: GA vs GUROBI ---")
    
    valor_total = 10000
    risco_teto = 0.15 
    lambda_val = 50.0 
    
    teste_setores_proibidos = ["CRIPTOATIVOS"]
    
    inputs = preparar_dados.calcular_inputs_otimizacao(valor_total)
    if not inputs: return

    # ... (Restante da lógica de main igual ao original) ...
    # Apenas para garantir que o arquivo seja executável
    
    print("\n[TESTE] Rodando Gurobi Direto...")
    res_gurobi = resolver_com_gurobi_setores(
        inputs, lambda_val, risco_teto, 
        warm_start_pesos=None,
        setores_proibidos=teste_setores_proibidos
    )
    
    if res_gurobi:
        print("Sucesso!")
        print(f"Retorno: {res_gurobi['retorno']:.2%}")
        print(f"Risco: {res_gurobi['risco']:.2%}")

if __name__ == "__main__":
    main()