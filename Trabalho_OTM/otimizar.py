import pandas as pd
from pymoo.optimize import minimize
import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.repair import Repair
from pymoo.termination.default import DefaultSingleObjectiveTermination

# --- IMPORTS DO PROJETO ---
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

# ==============================================================================
# CLASSE DE REPARO CUSTOMIZADA (CORRIGIDA)
# ==============================================================================
class SectorCapRepair(Repair):
    def __init__(self, mapa_setores, nomes_ativos, teto_setor, xu):
        """
        Inicializa o reparador com os índices dos setores para performance rápida.
        """
        # --- CORREÇÃO AQUI: Inicializa a classe pai do Pymoo ---
        super().__init__() 
        
        self.teto_setor = teto_setor
        self.xu = xu # Limites superiores individuais (Liquidez e Teto Ativo)
        
        # Pré-processamento: Mapear quais colunas (índices) pertencem a qual setor
        self.indices_setores = [] 
        if mapa_setores:
            for setor, lista_ativos_setor in mapa_setores.items():
                # Encontra os índices numéricos dos ativos deste setor
                idxs = [i for i, nome in enumerate(nomes_ativos) if nome in lista_ativos_setor]
                if idxs:
                    self.indices_setores.append(idxs)

    def _do(self, problem, X, **kwargs):
        # 1. LIMPEZA BÁSICA (NaNs e Valores Infinitos)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 2. LIMPEZA DE RUÍDO (Remove pesos muito pequenos)
        X[X < 0.005] = 0.0
        
        # 3. RESPEITAR LIMITES INDIVIDUAIS (Liquidez + Teto Usuário)
        if self.xu is not None:
             X = np.minimum(X, self.xu)

        # 4. NORMALIZAÇÃO (Apenas se estourar o orçamento de 100%)
        somas = X.sum(axis=1, keepdims=True)
        
        # Se a soma for > 1.0, normalizamos para baixo (divide pela soma)
        # Se a soma for < 1.0, NÃO normalizamos para cima, pois isso violaria o teto individual (xu)
        # O GA vai aprender a preencher o espaço se for vantajoso.
        mask_estouro_orcamento = somas > 1.0
        fatores = np.ones_like(somas)
        np.putmask(fatores, mask_estouro_orcamento, 1.0 / (somas + 1e-9))
        X = X * fatores

        # 5. APLICAÇÃO DO TETO POR SETOR (Corte)
        if self.teto_setor < 0.999:
            for idxs in self.indices_setores:
                # Soma os pesos deste setor para cada indivíduo da população
                soma_setor = X[:, idxs].sum(axis=1, keepdims=True)
                
                # Identifica quais indivíduos estouraram o teto neste setor
                mask_estouro = soma_setor > self.teto_setor
                
                # Calcula o fator de redução
                fatores = np.ones_like(soma_setor)
                np.putmask(fatores, mask_estouro, self.teto_setor / (soma_setor + 1e-9))
                
                # Reduz os pesos dos ativos desse setor
                X[:, idxs] *= fatores
        
        # 6. LIMPEZA FINAL (Garante que cortes não geraram resíduos < 0.5%)
        X[X < 0.005] = 0.0

        # 7. RETORNO (Pode somar < 1.0, gerando Caixa)
        return X

# ==============================================================================
# FUNÇÃO PRINCIPAL DE EXECUÇÃO
# ==============================================================================
def rodar_otimização(inputs, risco_maximo_usuario, lambda_aversao_risco, 
                     setores_proibidos=None, 
                     teto_maximo_ativo=0.30, 
                     teto_maximo_setor=1.0):
    
    # 1. Extração dos Inputs
    retornos_medios = inputs['retornos_medios']
    matriz_cov = inputs['matriz_cov']
    nomes_dos_ativos = inputs['nomes_dos_ativos']
    
    valor_investido = inputs.get('valor_total_investido', 0.0)
    vetor_pvp = inputs['vetor_pvp']
    vetor_cvar = inputs['vetor_cvar']
    volume_medio = inputs['volume_medio']
    
    # Obtém o mapa de setores para passar ao Repair
    mapa_setores = config.obter_mapa_setores_ativos()

    print("\n[GA] Inicializando Modelo Multiobjetivo...")
    
    # 2. Instancia o Problema (Passando os novos tetos)
    problema = modelo_problema_setor.OtimizacaoPortfolio(
        retornos_medios=retornos_medios,
        matriz_cov=matriz_cov,
        vetor_pvp=vetor_pvp,
        vetor_cvar=vetor_cvar,
        volume_medio=volume_medio,
        valor_investido=valor_investido,
        risco_maximo_usuario=risco_maximo_usuario,
        lambda_aversao_risco=lambda_aversao_risco,
        nomes_ativos=nomes_dos_ativos,
        mapa_setores=mapa_setores,
        setores_proibidos=setores_proibidos,
        teto_maximo_ativo=teto_maximo_ativo, 
        teto_maximo_setor=teto_maximo_setor  
    )
    
    print(f"[GA] Configurando restrições...")
    if setores_proibidos: print(f"   > Setores Proibidos: {setores_proibidos}")
    print(f"   > Teto por Ativo: {teto_maximo_ativo:.1%}")
    print(f"   > Teto por Setor: {teto_maximo_setor:.1%}")
    print()
        
    print(f"[GA] Rodando Evolução ({NUM_GERACOES} gerações)...")
    
    # 3. Configura o Algoritmo com o novo Repair
    algoritmo = GA(
        pop_size=POPULACAO_SIZE,
        eliminate_duplicates=True,
        repair=SectorCapRepair(
            mapa_setores=mapa_setores, 
            nomes_ativos=nomes_dos_ativos, 
            teto_setor=teto_maximo_setor,
            xu=problema.xu 
        )
    )

    # 4. Executa a Otimização
    res = minimize(
        problem=problema,
        algorithm=algoritmo,
        termination=TERMINATION,
        seed=1,
        verbose=False 
    )

    # 5. Processa Resultados
    if res and res.X is not None:
        print(f"Otimização concluída após {res.algorithm.n_gen} gerações.")
        pesos_otimos = res.X
        utility_score = res.F[0]
        
        # Verifica se houve "Caixa" (Soma < 1.0)
        soma_pesos = np.sum(pesos_otimos)
        if soma_pesos < 0.99:
            print(f"[GA] Aviso: Restrições impediram 100% de alocação. Investido: {soma_pesos:.1%}")
        
        # Métricas Finais
        variancia = pesos_otimos.dot(matriz_cov).dot(pesos_otimos)
        risco_otimo = np.sqrt(variancia)
        retorno_otimo = pesos_otimos.dot(retornos_medios)
        pvp_final = pesos_otimos.dot(vetor_pvp)
        cvar_final = pesos_otimos.dot(vetor_cvar)
        sharpe_final = retorno_otimo / (risco_otimo + 1e-9)
        
        # Recalcula a função objetivo PADRÃO (sem Sharpe) para comparação justa com Gurobi
        # O GA usa Sharpe internamente para busca, mas reporta a métrica padrão Mean-Variance
        
        penalidade_caixa = config.PESO_PENALIZACAO_CAIXA * max(0.0, 1.0 - pesos_otimos.sum())
        
        utility_score_reporting = (lambda_aversao_risco * (risco_otimo ** 2)) - retorno_otimo \
                                  + (config.PESO_PVP * pvp_final) \
                                  + (config.PESO_CVAR * cvar_final) \
                                  + penalidade_caixa
        
        # DataFrame para CSV/Logs
        df_pesos = pd.DataFrame([pesos_otimos], columns=nomes_dos_ativos)
        df_objetivos = pd.DataFrame({
            'Risco_Alvo': [risco_maximo_usuario],
            'Risco_Encontrado_Anual': [risco_otimo],
            'Retorno_Encontrado_Anual': [retorno_otimo]
        })
        
        df_final = pd.concat([df_objetivos, df_pesos], axis=1)
        
        sufixo_setor = "_setores_restritos" if setores_proibidos else ""
        nome_arq = f"carteira_L{lambda_aversao_risco}_R{int(risco_maximo_usuario*100)}{sufixo_setor}.csv"
        
        return {
            "metricas": {
                "retorno_aa": retorno_otimo * 100,
                "risco_aa": risco_otimo * 100,
                "score": utility_score_reporting, # Métrica Padronizada
                "funcao_objetivo": utility_score, # Métrica Interna (com Sharpe)
                "lambda_risco": lambda_aversao_risco,
                "pvp_final": pvp_final,   
                "cvar_final": cvar_final,
                "sharpe": sharpe_final
            },
            "dataframe_resultado": df_final,
            "nome_arquivo_csv": nome_arq,
            "pesos_finais": pesos_otimos,
            "risco_final": risco_otimo,
            "retorno_final": retorno_otimo,
            "funcao_objetivo": utility_score
        }
            
    else:
        print("\nALERTA: Otimização não convergiu para uma solução viável.")
        return None