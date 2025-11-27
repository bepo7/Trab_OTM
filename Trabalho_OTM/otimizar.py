import pandas as pd
from pymoo.optimize import minimize
import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.repair import Repair
from pymoo.termination.default import DefaultSingleObjectiveTermination

# --- IMPORTS ---
import modelo_problema_setor
import config

POPULACAO_SIZE = 100
NUM_GERACOES = 1500

# --- CRITÉRIO DE PARADA ---
TERMINATION = DefaultSingleObjectiveTermination(
    ftol=1e-9,         
    period=100,        
    n_max_gen=NUM_GERACOES, 
    xtol=1e-20,        
    cvtol=1e-20        
)

# --- CLASSE DE REPARO (Normalização) ---
class MinimumWeightRepair(Repair):
    def _do(self, problem, X, **kwargs):
        # Garante que pesos muito pequenos virem zero e a soma seja 1
        limiar = 0.005 
        X[X < limiar] = 0.0
        
        # Se o modelo tiver limites superiores (xu) iguais a 0 (setores proibidos),
        # garantimos que o reparo respeite isso forçando 0 novamente.
        if problem.xu is not None:
             X = np.minimum(X, problem.xu)

        somas = X.sum(axis=1, keepdims=True)
        somas[somas == 0] = 1.0 
        X = X / somas
        return X

def rodar_otimização(inputs, risco_maximo_usuario, lambda_aversao_risco, setores_proibidos=None):
    # 1. Extração dos Inputs
    retornos_medios = inputs['retornos_medios']
    matriz_cov = inputs['matriz_cov']
    nomes_dos_ativos = inputs['nomes_dos_ativos']
    
    # IMPORTANTE: Resgatando o valor investido e novos vetores
    valor_investido = inputs.get('valor_total_investido', 0.0)
    vetor_pvp = inputs['vetor_pvp']
    vetor_cvar = inputs['vetor_cvar']
    volume_medio = inputs['volume_medio']
    
    # Obtém o mapa de setores
    mapa_setores = config.obter_mapa_setores_ativos()

    print("\n[GA] Inicializando Modelo Multiobjetivo (Retorno, Risco, P/VP, CVaR)...")
    
    # Instancia o modelo atualizado com os novos parâmetros
    problema = modelo_problema_setor.OtimizacaoPortfolio(
        retornos_medios=retornos_medios,
        matriz_cov=matriz_cov,
        vetor_pvp=vetor_pvp,     # <--- Novo
        vetor_cvar=vetor_cvar,   # <--- Novo
        volume_medio=volume_medio,   # <--- Passando volume
        valor_investido=valor_investido,
        risco_maximo_usuario=risco_maximo_usuario,
        lambda_aversao_risco=lambda_aversao_risco,
        nomes_ativos=nomes_dos_ativos,
        mapa_setores=mapa_setores,
        setores_proibidos=setores_proibidos
    )

    print(f"[GA] Rodando Evolução. Parada: 100 gerações ESTÁVEIS ou {NUM_GERACOES} gerações.")
    
    algoritmo = GA(
        pop_size=POPULACAO_SIZE,
        eliminate_duplicates=True,
        repair=MinimumWeightRepair()
    )

    res = minimize(
        problem=problema,
        algorithm=algoritmo,
        termination=TERMINATION,
        seed=1,
        verbose=False 
    )

    if res and res.X is not None:
        print(f"Otimização concluída após {res.algorithm.n_gen} gerações.")
        pesos_otimos = res.X
        utility_score = res.F[0]
        
        # Recalcula métricas para exibição (usando apenas média/variância clássica)
        variancia = pesos_otimos.dot(matriz_cov).dot(pesos_otimos)
        risco_otimo = np.sqrt(variancia)
        retorno_otimo = pesos_otimos.dot(retornos_medios)
        
        print(f"  Valor da função objetivo: {utility_score:.4f}")
        print(f"  Risco Encontrado: {risco_otimo:.2%} (Teto: {risco_maximo_usuario:.2%})")
        print(f"  Retorno Esperado: {retorno_otimo:.2%}")
        
        # --- CRIAÇÃO DO DATAFRAME (Necessário para o plot.py) ---
        df_pesos = pd.DataFrame([pesos_otimos], columns=nomes_dos_ativos)
        df_objetivos = pd.DataFrame({
            'Risco_Alvo': [risco_maximo_usuario],
            'Risco_Encontrado_Anual': [risco_otimo],
            'Retorno_Encontrado_Anual': [retorno_otimo]
        })
        
        df_final = pd.concat([df_objetivos, df_pesos], axis=1)
        
        # Opcional: Gerar nome de arquivo
        sufixo_setor = "_setores_restritos" if setores_proibidos else ""
        nome_arq = f"carteira_L{lambda_aversao_risco}_R{int(risco_maximo_usuario*100)}{sufixo_setor}.csv"
        
        return {
            "metricas": {
                "valor_investido": valor_investido,
                "retorno_esperado": retorno_otimo,
                "risco_esperado": risco_otimo,
                "funcao_objetivo": utility_score,
                "lambda_risco": lambda_aversao_risco
            },
            "dataframe_resultado": df_final, # <--- Corrigido (agora é um DF real)
            "nome_arquivo_csv": nome_arq,
            "pesos_finais": pesos_otimos,
            "risco_final": risco_otimo,
            "retorno_final": retorno_otimo,
            "funcao_objetivo": utility_score
        }
            
    else:
        print("\nALERTA: Otimização não convergiu para uma solução viável.")
        return None