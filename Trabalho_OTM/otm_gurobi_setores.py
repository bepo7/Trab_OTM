import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import math

import preparar_dados
import otimizar
import config

# Função segura para converter valores para float
def safe_float(val):
    try:
        val = float(val)
        if math.isnan(val) or math.isinf(val): return 0.0
        return val
    except: return 0.0

def limpar_string(s):
    return str(s).strip().upper()

# Função principal para resolver o problema com Gurobi e restrições setoriais
def resolver_com_gurobi_setores(inputs, lambda_risk, risco_max_usuario, 
                                warm_start_pesos, setores_proibidos,
                                teto_maximo_ativo=0.30,
                                teto_maximo_setor=1.0,
                                max_ativos_carteira=15, 
                                max_ativos_setor=4,     
                                verbose=True):          
    
    # Variável mínima de peso para considerar compra
    MIN_PESO_SE_COMPRAR = 0.005
 
    # 1. Extração dos Inputs
    retornos = inputs['retornos_medios'].values
    cov_matrix = inputs['matriz_cov'].values
    vals_pvp = inputs['vetor_pvp'].values
    vals_cvar = inputs['vetor_cvar'].values
    nomes_ativos = inputs['nomes_dos_ativos']
    n_ativos = len(retornos)

    volume_medio = inputs['volume_medio']
    valor_investido = inputs['valor_total_investido'] or 1.0
    
    # Obtém os preços atuais (últimos preços)
    if 'ultimos_precos' in inputs:
        precos_atuais = inputs['ultimos_precos']
    else:
        print("⚠️ AVISO: Preços atuais não encontrados. Usando fallback de R$ 10,00.")
        precos_atuais = pd.Series([10.0] * n_ativos, index=nomes_ativos)

    if verbose:
        print(f"\n[GUROBI] Iniciando... Max Global: {max_ativos_carteira} | Max/Setor: {max_ativos_setor}")

    # 2. Mapeamento de Setores
    mapa_setores = config.obter_mapa_setores_ativos()
    indices_por_setor = {}
    ativo_para_setor = {}

    # Faz o mapeamento reverso de ativo para setor
    for setor, lista_ativos in mapa_setores.items():
        if setor not in indices_por_setor: indices_por_setor[setor] = []
        for ativo in lista_ativos:
            ativo_limpo = limpar_string(ativo)
            ativo_para_setor[ativo_limpo] = setor

    # Preenche os índices por setor
    for i, ticker in enumerate(nomes_ativos):
        ticker_limpo = limpar_string(ticker)
        if ticker_limpo in ativo_para_setor:
            setor = ativo_para_setor[ticker_limpo]
            indices_por_setor[setor].append(i)

    # Constrói o conjunto de ativos proibidos com base nos setores proibidos
    ativos_proibidos_set = set()
    if setores_proibidos:
        for setor in setores_proibidos:
            if setor in mapa_setores:
                ativos_proibidos_set.update([limpar_string(a) for a in mapa_setores[setor]])

    # 3. Construção do Modelo Gurobi
    model = gp.Model("Portfolio_Dinamico")
    model.setParam('OutputFlag', 0) 
    
    # Variáveis de Decisão
    pesos = []
    vars_lotes = []
    vars_binarias = [] 

    # Loop para criar variáveis por ativo
    for i, ticker in enumerate(nomes_ativos):
        ticker_limpo = limpar_string(ticker)
        
        preco_unitario = safe_float(precos_atuais.get(ticker, 0.0))
        custo_acao = preco_unitario
        vol = safe_float(volume_medio.get(ticker, 0.0))
        
        # Cálculo do teto financeiro baseado no menor entre teto e liquidez
        teto_financeiro_ativo = min(valor_investido * teto_maximo_ativo, 0.1 * vol)
        if ticker_limpo in ativos_proibidos_set: teto_financeiro_ativo = 0.0
            
        if custo_acao <= 0.01: 
            max_unidades = 0
        else:
            max_unidades = int(teto_financeiro_ativo / custo_acao)
        
        # Cálculo da quantidade mínima de cotas para comprar
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
            
            if peso_ga > 1e-6: 
                valor_sugerido = peso_ga * valor_investido
                
                # Cálculo da quantidade sugerida com base no preço atual
                if custo_acao > 0.01:
                    qtd_sugerida = int(valor_sugerido / custo_acao)
                    
                    if qtd_sugerida > max_unidades: qtd_sugerida = max_unidades
                    
                    # Define o valor inicial para a variável de quantidade
                    n_cotas.Start = qtd_sugerida

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

    # 4. Função Objetivo e Restrições

    # Restrição de Orçamento que pode ser menor que 1.0 (permitindo Caixa)
    expr_ret = gp.quicksum(pesos[i] * retornos[i] for i in range(n_ativos))
    # Restrição de Risco que deve ser menor que o máximo do usuário
    expr_var = gp.quicksum(pesos[i] * cov_matrix[i, j] * pesos[j] for i in range(n_ativos) for j in range(n_ativos))
    # Restrições de Teto Setorial que limitam o peso total por setor
    expr_pvp = gp.quicksum(pesos[i] * vals_pvp[i] for i in range(n_ativos))
    # Restrição de CVaR que deve ser menor que o máximo do usuário
    expr_cvar = gp.quicksum(pesos[i] * vals_cvar[i] for i in range(n_ativos))
    # Restrição de Soma dos Pesos que deve ser menor que 1.0
    expr_soma_pesos = gp.quicksum(pesos)
    
    # Função Objetivo (Multiobjetivo Scalarizado) dado pelo usuário
    # Minimizar: (Risco * Lambda) - Retorno + Custo P/VP + Custo CVaR + Penalidade Caixa
    obj = (lambda_risk * expr_var) - expr_ret + (config.PESO_PVP * expr_pvp) + (config.PESO_CVAR * expr_cvar) + (config.PESO_PENALIZACAO_CAIXA * (1.0 - expr_soma_pesos))
    model.setObjective(obj, GRB.MINIMIZE)
    
    model.addConstr(expr_soma_pesos <= 1.0, "orcamento")
    model.addConstr(expr_var <= risco_max_usuario ** 2, "Risco")
    
    # Restrições de Teto Setorial
    if teto_maximo_setor < 0.999:
        for setor, idxs in indices_por_setor.items():
            if idxs:
                model.addConstr(gp.quicksum(pesos[i] for i in idxs) <= teto_maximo_setor, f"TetoFin_{setor}")

    # Restrições de Cardinalidade

    # 1. Global
    model.addConstr(gp.quicksum(vars_binarias) <= max_ativos_carteira, "Card_Global")

    # 2. Por Setor
    for setor, idxs in indices_por_setor.items():
        if idxs:
            model.addConstr(gp.quicksum(vars_binarias[i] for i in idxs) <= max_ativos_setor, f"Card_Setor_{setor}")

    model.optimize()
    
    # 5. Extração dos Resultados
    if model.Status == GRB.OPTIMAL:
        w_otimo = np.zeros(n_ativos)
        lotes_otimos = np.zeros(n_ativos)
        
        for i in range(n_ativos):
            qtd = vars_lotes[i].X
            preco = safe_float(precos_atuais.get(nomes_ativos[i], 0.0))
            if valor_investido > 0:
                w_otimo[i] = (qtd * preco) / valor_investido
            lotes_otimos[i] = qtd
        
        # Cálculo das métricas finais
        ret_final = np.dot(w_otimo, retornos)
        var_final = np.dot(w_otimo, np.dot(cov_matrix, w_otimo))
        pvp_final = np.dot(w_otimo, vals_pvp)
        cvar_final = np.dot(w_otimo, vals_cvar)
        
        investido_real = w_otimo.sum() * valor_investido

        # Garante que o investido real não exceda o valor investido (por precaução)
        if investido_real > valor_investido:
            investido_real = valor_investido
        
        sobra = valor_investido - investido_real

        qtd_ativos_selecionados = int(sum(vars_binarias[i].X for i in range(n_ativos)))
        
        # Impressão dos resultados
        if verbose:
            print(f"   > [SUCESSO] Inv: R$ {investido_real:.2f} | Sobra: R$ {sobra:.2f} | Ativos: {qtd_ativos_selecionados}")
        if verbose:
            print()
        
        # 6. Retorno dos Resultados
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
        if verbose:
            print(f"[GUROBI] Falha. Status: {model.Status}")
        return None