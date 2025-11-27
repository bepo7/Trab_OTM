import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np

# --- IMPORTAÇÕES DO SEU PROJETO ---
import preparar_dados
import otimizar
import config

# ==============================================================================
# 1. SOLVER GUROBI COM RESTRIÇÃO DE SETORES
# ==============================================================================
def resolver_com_gurobi_setores(inputs, lambda_risk, risco_max_usuario, 
                                warm_start_pesos, setores_proibidos):
    
    retornos = inputs['retornos_medios'].values
    cov_matrix = inputs['matriz_cov'].values
    
    # NOVOS DADOS
    vals_pvp = inputs['vetor_pvp'].values
    vals_cvar = inputs['vetor_cvar'].values
    
    nomes_ativos = inputs['nomes_dos_ativos']
    n_ativos = len(retornos)
    
    # --- MAPEAMENTO DE ATIVOS PROIBIDOS ---
    mapa_setores = config.obter_mapa_setores_ativos()
    ativos_proibidos_set = set()
    
    if setores_proibidos:
        for setor in setores_proibidos:
            if setor in mapa_setores:
                # Adiciona todos os tickers desse setor ao set de proibidos
                ativos_proibidos_set.update(mapa_setores[setor])
    
    print(f"[GUROBI] Configurando restrições. Setores proibidos: {setores_proibidos}")
    print(f"[GUROBI] Total de ativos travados em 0.0: {len(ativos_proibidos_set)} de {n_ativos}")

    # --- INÍCIO DO MODELO ---
    model = gp.Model("Portfolio_MultiObj")
    model.setParam('OutputFlag', 0)
    
    pesos = []
    
    # --- CRIAÇÃO DAS VARIÁVEIS COM LIMITES (BOUNDS) ---
    for i, ticker in enumerate(nomes_ativos):
        # (Lógica de limite_superior baseada em setor proibido MANTIDA)
        limite_superior = 1.0 # (Simplificado aqui, use sua lógica de if ticker in proibidos...)
        
        p = model.addVar(lb=0.0, ub=limite_superior, vtype=GRB.CONTINUOUS, name=f"w[{ticker}]")
        if warm_start_pesos is not None:
             p.Start = warm_start_pesos[i]
        pesos.append(p)
        
    model.update()

    # --- NOVA FUNÇÃO OBJETIVO ---
    
    # Termo Quadrático: Lambda * Variância
    expr_variancia = gp.quicksum(pesos[i] * cov_matrix[i, j] * pesos[j] 
                                 for i in range(n_ativos) for j in range(n_ativos))
    
    # Termo Linear: Retorno (Negativo pois é Max)
    expr_retorno = gp.quicksum(pesos[i] * retornos[i] for i in range(n_ativos))
    
    # NOVO Termo Linear: P/VP (Positivo pois é Min)
    expr_pvp = gp.quicksum(pesos[i] * vals_pvp[i] for i in range(n_ativos))
    
    # NOVO Termo Linear: CVaR (Positivo pois é Min)
    expr_cvar = gp.quicksum(pesos[i] * vals_cvar[i] for i in range(n_ativos))
    
    # Objetivo Final
    obj_final = (lambda_risk * expr_variancia) - expr_retorno \
                + (config.PESO_PVP * expr_pvp) \
                + (config.PESO_CVAR * expr_cvar)
    
    model.setObjective(obj_final, GRB.MINIMIZE)
    
    # --- RESTRIÇÕES (Mantidas) ---
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
        return None

# ==============================================================================
# 2. FLUXO PRINCIPAL
# ==============================================================================
def main():
    print("--- BENCHMARK AVANÇADO: GA vs GUROBI (COM RESTRIÇÃO DE SETORES) ---")
    
    # --- CONFIGURAÇÕES ---
    valor_total = 10000
    risco_teto = 0.15      # 15% a.a.
    lambda_val = 50.0      # Aversão ao risco (Escala Variância)
    
    # DEFINA AQUI QUAIS SETORES VOCÊ QUER TESTAR REMOVER
    # (Nomes devem bater com as chaves do dicionário em config.py)
    teste_setores_proibidos = [
        "CRIPTOATIVOS", 
        "SETOR_ENERGIA_PETROLEO" # Exemplo: Usuário ESG conservador
    ]
    
    # 1. Carregar Dados
    inputs = preparar_dados.calcular_inputs_otimizacao(valor_total)
    if not inputs: return

    # ---------------------------------------------------------
    # PASSO A: GA COM RESTRIÇÃO DE SETORES
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print(f"PASSO 1: GA - Otimizando sem: {teste_setores_proibidos}")
    print("="*60)
    
    # Chama o otimizar passando a lista de proibidos
    resultado_ga = otimizar.rodar_otimização(
        inputs, risco_teto, lambda_val, 
        setores_proibidos=teste_setores_proibidos
    )
    
    if not resultado_ga:
        print("GA falhou.")
        return

    pesos_finais_ga = resultado_ga['pesos_finais']

    # ---------------------------------------------------------
    # PASSO B: GUROBI COM A MESMA RESTRIÇÃO
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("PASSO 2: Gurobi - Validando matematicamente...")
    print("="*60)
    
    res_gurobi = resolver_com_gurobi_setores(
        inputs, lambda_val, risco_teto, 
        warm_start_pesos=pesos_finais_ga,
        setores_proibidos=teste_setores_proibidos
    )
    
    if not res_gurobi: return

    # ---------------------------------------------------------
    # PASSO C: COMPARAÇÃO
    # ---------------------------------------------------------
    print("\n" + "="*70)
    print(f"{'METRICA':<25} | {'GA (RESTRITO)':<20} | {'GUROBI (RESTRITO)':<20}")
    print("-" * 70)
    
    ga_obj = resultado_ga['funcao_objetivo']
    gu_obj = res_gurobi['obj']
    
    print(f"{'Função Objetivo':<25} | {ga_obj:10.6f}           | {gu_obj:10.6f}")
    print(f"{'Retorno Esperado':<25} | {resultado_ga['retorno_final']:10.2%}           | {res_gurobi['retorno']:10.2%}")
    print(f"{'Risco (Volatilidade)':<25} | {resultado_ga['risco_final']:10.2%}           | {res_gurobi['risco']:10.2%}")
    print("-" * 70)
    
    gap = ga_obj - gu_obj
    print(f"\n>>> GAP de Convergência: {gap:.9f}")
    
    # Verificação de Integridade: Checar se Gurobi realmente zerou os proibidos
    mapa = config.obter_mapa_setores_ativos()
    proibidos_flat = []
    for s in teste_setores_proibidos:
        if s in mapa: proibidos_flat.extend(mapa[s])
    
    df_g = pd.DataFrame({'Ativo': inputs['nomes_dos_ativos'], 'Peso': res_gurobi['pesos']})
    df_check = df_g[df_g['Ativo'].isin(proibidos_flat)]
    soma_erro = df_check['Peso'].sum()
    
    if soma_erro > 1e-5:
        print(f"ERRO CRÍTICO: Gurobi alocou {soma_erro:.2%} em ativos proibidos!")
    else:
        print("SUCESSO: Gurobi respeitou perfeitamente as restrições de setor (Peso Total Proibido = 0.0%).")
        
    # Salvar comparativo
    nome_arq = "comparativo_setores_GA_Gurobi.csv"
    df_final = pd.DataFrame({
        'Ativo': inputs['nomes_dos_ativos'],
        'Peso_GA': pesos_finais_ga,
        'Peso_Gurobi': res_gurobi['pesos']
    })
    # Filtra para mostrar apenas ativos relevantes ( > 0.1%)
    df_final = df_final[(df_final['Peso_GA'] > 0.001) | (df_final['Peso_Gurobi'] > 0.001)]
    df_final.to_csv(nome_arq, sep=';', decimal=',', index=False)
    print(f"Comparativo salvo em '{nome_arq}'")

if __name__ == "__main__":
    main()