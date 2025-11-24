import pandas as pd
from pymoo.optimize import minimize
import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.repair import Repair

# --- IMPORTA TERMINADOR PADRÃO ---
from pymoo.termination.default import DefaultSingleObjectiveTermination

# --- IMPORTA OS DOIS MODELOS ---
import modelo_problema
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

# --- CLASSE DE REPARO ---
class MinimumWeightRepair(Repair):
    def _do(self, problem, X, **kwargs):
        limiar = 0.005  # 0.5%
        X[X < limiar] = 0.0
        somas = X.sum(axis=1, keepdims=True)
        somas[somas == 0] = 1.0 
        X = X / somas
        return X

def rodar_otimização(inputs, risco_maximo_usuario, lambda_aversao_risco, setores_proibidos=None):
    retornos_medios = inputs['retornos_medios']
    matriz_cov = inputs['matriz_cov']
    nomes_dos_ativos = inputs['nomes_dos_ativos']
    valor_investido = inputs.get('valor_total_investido', 0.0) # Recupera o valor investido
    
    if setores_proibidos and len(setores_proibidos) > 0:
        print(f"\n[MODO AVANÇADO] Aplicando restrições para zerar {len(setores_proibidos)} setores.")
        problema = modelo_problema_setor.OtimizacaoPortfolioSetor(
            retornos_medios=retornos_medios,
            matriz_cov=matriz_cov,
            risco_maximo_usuario=risco_maximo_usuario,
            lambda_aversao_risco=lambda_aversao_risco,
            nomes_ativos_ordenados=nomes_dos_ativos,
            mapa_setores=config.obter_mapa_setores_ativos(),
            setores_proibidos=setores_proibidos
        )
    else:
        print("\n[MODO PADRÃO] Otimização sem restrições setoriais.")
        problema = modelo_problema.OtimizacaoPortfolio(
            retornos_medios,
            matriz_cov,
            risco_maximo_usuario,
            lambda_aversao_risco
        )

    print(f"[GA] Iniciando GA. Parada: 100 gerações ESTÁVEIS ou {NUM_GERACOES} gerações.")
    
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
        verbose=True
    )

    if res and res.X is not None:
        print(f"\nOtimização concluída após {res.algorithm.n_gen} gerações.")
        pesos_otimos = res.X
        utility_score = res.F[0]
        
        variancia = pesos_otimos.dot(matriz_cov).dot(pesos_otimos)
        risco_otimo = np.sqrt(variancia)
        retorno_otimo = pesos_otimos.dot(retornos_medios)
        
        print(f"  Valor da função objetivo: {utility_score:.4f}")
        print(f"  Risco Encontrado: {risco_otimo:.2%} (Teto: {risco_maximo_usuario:.2%})")
        print(f"  Retorno Esperado: {retorno_otimo:.2%}")
        
        df_pesos = pd.DataFrame([pesos_otimos], columns=nomes_dos_ativos)
        df_objetivos = pd.DataFrame({
            'Risco_Alvo': [risco_maximo_usuario],
            'Risco_Encontrado_Anual': [risco_otimo],
            'Retorno_Encontrado_Anual': [retorno_otimo]
        })
        
        df_final = pd.concat([df_objetivos, df_pesos], axis=1)
        sufixo_setor = "_setores_restritos" if setores_proibidos else ""
        nome_arq = f"carteira_L{lambda_aversao_risco}_R{int(risco_maximo_usuario*100)}{sufixo_setor}.csv"
        
        df_final.to_csv(nome_arq, index=False, sep=';', decimal=',')
        
        # --- CORREÇÃO AQUI: ESTRUTURA PARA O FRONTEND ---
        return {
            "metricas": {
                "valor_investido": valor_investido,
                "retorno_esperado": retorno_otimo,
                "risco_esperado": risco_otimo,
                "funcao_objetivo": utility_score,
                "lambda_risco": lambda_aversao_risco
            },
            "dataframe_resultado": df_final, 
            "nome_arquivo_csv": nome_arq,
            "pesos_finais": pesos_otimos,
            # Mantendo chaves planas por segurança (compatibilidade)
            "risco_final": risco_otimo,
            "retorno_final": retorno_otimo,
            "funcao_objetivo": utility_score
        }
            
    else:
        print("\nALERTA: Otimização não convergiu para uma solução viável.")
        return None