import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np

# --- IMPORTAÇÕES DO SEU PROJETO ---
import preparar_dados
import otimizar  # Agora chamamos a execução completa dele
import config

# ==============================================================================
# 1. SOLVER GUROBI (COM RESTRIÇÃO DE NÃO-NEGATIVIDADE)
# ==============================================================================
def resolver_com_gurobi(inputs, lambda_risk, risco_max_usuario, warm_start_pesos):
    """
    Resolve usando Gurobi (Solver Exato).
    Aceita 'warm_start_pesos' para tentar melhorar uma solução existente.
    """
    retornos = inputs['retornos_medios'].values
    cov_matrix = inputs['matriz_cov'].values
    n_ativos = len(retornos)
    
    model = gp.Model("Portfolio_Mean_Variance")
    model.setParam('OutputFlag', 0) # 0 = Silencioso
    
    # --- VARIÁVEIS ---
    # lb=0.0 garante pesos positivos
    pesos = model.addVars(n_ativos, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="w")
    
    # --- WARM START (PONTO DE PARTIDA) ---
    # Aqui injetamos a solução final do GA
    if warm_start_pesos is not None:
        for i in range(n_ativos):
            pesos[i].Start = warm_start_pesos[i]
            
    # --- FUNÇÃO OBJETIVO (Média-Variância) ---
    # Min (Lambda * w'Cov'w - w'Mu)
    expr_retorno = gp.quicksum(pesos[i] * retornos[i] for i in range(n_ativos))
    expr_variancia = gp.quicksum(pesos[i] * cov_matrix[i, j] * pesos[j] 
                                 for i in range(n_ativos) 
                                 for j in range(n_ativos))
    
    model.setObjective((lambda_risk * expr_variancia) - expr_retorno, GRB.MINIMIZE)
    
    # --- RESTRIÇÕES ---
    # 1. Orçamento: Soma(w) == 1
    model.addConstr(gp.quicksum(pesos[i] for i in range(n_ativos)) == 1.0, "Orcamento")
    
    # 2. Teto de Risco: Variância <= (RiscoMax)^2
    limite_variancia = risco_max_usuario ** 2
    model.addConstr(expr_variancia <= limite_variancia, "Teto_Risco_Quadratico")
    
    # --- OTIMIZAR ---
    model.optimize()
    
    if model.Status == GRB.OPTIMAL:
        w_otimo = np.array([pesos[i].X for i in range(n_ativos)])
        
        # Recalcula métricas finais
        ret_final = np.dot(w_otimo, retornos)
        var_final = np.dot(w_otimo, np.dot(cov_matrix, w_otimo))
        
        return {
            'pesos': w_otimo,
            'obj': model.ObjVal,
            'retorno': ret_final,
            'risco': np.sqrt(var_final)
        }
    else:
        print(f"Gurobi falhou ou é inviável. Status Code: {model.Status}")
        return None

# ==============================================================================
# 2. FLUXO PRINCIPAL DE COMPARAÇÃO
# ==============================================================================
def main():
    print("--- VALIDAR CONVERGÊNCIA: GA COMPLETO vs GUROBI ---")
    print("Este script roda o GA até o fim e depois tenta melhorar o resultado com Matemática Exata.")
    
    # Configurações
    valor_total = 10000
    risco_teto = 0.15   # 15% a.a.
    lambda_val = 50.0   # Lambda ajustado para Variância
    
    # 1. Carregar Dados
    inputs = preparar_dados.calcular_inputs_otimizacao(valor_total)
    if not inputs: return

    # ---------------------------------------------------------
    # PASSO A: RODAR O ALGORITMO GENÉTICO COMPLETO
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("PASSO 1: Executando Algoritmo Genético (Meta-heurística)...")
    print("="*50)
    
    # Chama a função original do seu projeto
    resultado_ga = otimizar.rodar_otimização(inputs, risco_teto, lambda_val)
    
    if not resultado_ga:
        print("GA falhou em encontrar solução.")
        return

    pesos_finais_ga = resultado_ga['pesos_finais']

    # ---------------------------------------------------------
    # PASSO B: RODAR GUROBI (COM A SOLUÇÃO DO GA COMO START)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("PASSO 2: Refinando com Gurobi (Solver Exato)...")
    print("="*50)
    
    res_gurobi = resolver_com_gurobi(inputs, lambda_val, risco_teto, warm_start_pesos=pesos_finais_ga)
    
    if not res_gurobi: return

    # ---------------------------------------------------------
    # PASSO C: RELATÓRIO DE CONVERGÊNCIA
    # ---------------------------------------------------------
    print("\n" + "="*65)
    print(f"{'METRICA':<25} | {'GA (FINAL)':<20} | {'GUROBI (REFINADO)':<20}")
    print("-" * 65)
    
    # Extraindo valores GA
    ga_obj = resultado_ga['funcao_objetivo']
    ga_ret = resultado_ga['retorno_final']
    ga_ris = resultado_ga['risco_final']
    
    # Extraindo valores Gurobi
    gu_obj = res_gurobi['obj']
    gu_ret = res_gurobi['retorno']
    gu_ris = res_gurobi['risco']
    
    print(f"{'Função Objetivo (Min)':<25} | {ga_obj:10.6f}           | {gu_obj:10.6f}")
    print(f"{'Retorno Esperado':<25} | {ga_ret:10.2%}           | {gu_ret:10.2%}")
    print(f"{'Risco (Volatilidade)':<25} | {ga_ris:10.2%}           | {gu_ris:10.2%}")
    print("-" * 65)
    
    gap = ga_obj - gu_obj
    
    # Interpretação do Resultado
    print(f"\n>>> GAP de Convergência: {gap:.9f}")
    
    if gap < 1e-5:
        print(">>> CONCLUSÃO: O GA convergiu perfeitamente para o Ótimo Global! Parabéns.")
    elif gap < 1e-3:
        print(">>> CONCLUSÃO: O GA chegou muito perto (Ótimo Local de alta qualidade). Aceitável.")
    else:
        print(">>> CONCLUSÃO: O Gurobi conseguiu melhorar significativamente a carteira.")
        print("               Tente aumentar o número de gerações ou o tamanho da população do GA.")
    
    print("="*65)
    
    # Salvar Comparativo de Pesos
    df = pd.DataFrame({
        'Ativo': inputs['nomes_dos_ativos'],
        'Peso_GA': pesos_finais_ga,
        'Peso_Gurobi': res_gurobi['pesos']
    })
    
    # Mostrar apenas onde houve divergência relevante ou peso alto
    df_relevante = df[(df['Peso_GA'] > 1e-3) | (df['Peso_Gurobi'] > 1e-3)].copy()
    df_relevante['Diferenca'] = df_relevante['Peso_GA'] - df_relevante['Peso_Gurobi']
    
    nome_arq = "comparativo_final_GA_vs_Gurobi.csv"
    df_relevante.to_csv(nome_arq, index=False, sep=';', decimal=',')
    print(f"\nComparativo detalhado de pesos salvo em '{nome_arq}'")

if __name__ == "__main__":
    main()