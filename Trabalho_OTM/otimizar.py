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

MIN_PESO_ATIVO = 0.005  # 0.5%
TETO_GLOBAL_ATIVO = 0.30

class MinimumWeightRepair(Repair):
    def _do(self, problem, X, **kwargs):
        # 1. SEGURANÇA: Remove NaNs e Infinitos que possam ter vindo do Crossover
        # Se vier lixo, transforma em 0.0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 2. Limpeza de ruído (pesos muito pequenos viram zero)
        X[X < 0.005] = 0.0
        
        # 3. Garante limites superiores iniciais (setores proibidos e liquidez)
        if problem.xu is not None:
             X = np.minimum(X, problem.xu)

        # 4. Normalização Iterativa "Smart"
        for _ in range(20): # Tenta corrigir por 20 iterações
            somas = X.sum(axis=1, keepdims=True)
            
            # --- CORREÇÃO DO ERRO DE DIVISÃO ---
            # Adicionamos 1e-9 para nunca dividir por zero absoluto
            X = X / (somas + 1e-9)
            
            # Verifica se a normalização fez alguém estourar o limite (ex: 31%)
            if problem.xu is not None:
                X_clipped = np.minimum(X, problem.xu)
                
                # Se já está tudo dentro do limite, para
                if np.allclose(X, X_clipped, atol=1e-5):
                    X = X_clipped
                    break
                
                X = X_clipped
        
        # 5. SEGURANÇA FINAL: Normalização simples para garantir soma 1.0
        # (Mesmo que viole minimamente o teto em casas decimais, a soma 1 é prioritária pro solver não quebrar)
        somas_finais = X.sum(axis=1, keepdims=True)
        return X / (somas_finais + 1e-9)
    
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
    
    print(f"[GA] Configurando restrições...")
    print(f"   > Setores Proibidos: {setores_proibidos}")
    print(f"   > Mínimo de Entrada: {MIN_PESO_ATIVO:.1%}")
    print(f"   > Teto Global por Ativo: {TETO_GLOBAL_ATIVO:.1%}")
    print()
        
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
        pvp_final = pesos_otimos.dot(vetor_pvp)
        cvar_final = pesos_otimos.dot(vetor_cvar)
        
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
                "lambda_risco": lambda_aversao_risco,
                "pvp_final": pvp_final,   
                "cvar_final": cvar_final
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