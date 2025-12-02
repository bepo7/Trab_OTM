import pandas as pd
import numpy as np
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.repair import Repair
from pymoo.core.problem import Problem
from pymoo.termination.default import DefaultSingleObjectiveTermination

import config

# Parâmetros do Algoritmo Genético
POPULACAO_SIZE = 100
NUM_GERACOES = 1500

TERMINATION = DefaultSingleObjectiveTermination(
    ftol=1e-9,         
    period=100,        
    n_max_gen=NUM_GERACOES, 
    xtol=1e-20,        
    cvtol=1e-20        
)


class OtimizacaoPortfolio(Problem):
    def __init__(self, retornos_medios, matriz_cov, 
                 vetor_pvp, vetor_cvar, 
                 volume_medio, valor_investido,
                 risco_maximo_usuario, lambda_aversao_risco,
                 nomes_ativos=None, mapa_setores=None, setores_proibidos=None,
                 teto_maximo_ativo=0.30, teto_maximo_setor=1.0, verbose=True):
        
        self.retornos_medios = retornos_medios
        self.matriz_cov = matriz_cov
        self.vetor_pvp = vetor_pvp.values     
        self.vetor_cvar = vetor_cvar.values   
        
        self.risco_maximo_usuario = risco_maximo_usuario
        self.lambda_aversao_risco = lambda_aversao_risco
        
        self.teto_maximo_ativo = teto_maximo_ativo
        self.teto_maximo_setor = teto_maximo_setor
        n_ativos = len(retornos_medios)
        
        xl = np.full(n_ativos, 0.0)
        
        
        # 1. Cálculo do Teto por Liquidez
        vol_values = np.nan_to_num(volume_medio.values, nan=0.0)
        teto_financeiro_liquidez = 0.1 * vol_values
        
        # 2. Converte para teto em peso
        inv = valor_investido if valor_investido > 0 else 1.0
        max_peso_liquidez = teto_financeiro_liquidez / inv
        
        # 3. O teto final é o mínimo entre os tetos definidos
        xu = np.minimum(teto_maximo_ativo, max_peso_liquidez)
        xu = np.minimum(xu, teto_maximo_setor) 
        
        # Setores Proibidos: Zera os pesos desses ativos
        if setores_proibidos and nomes_ativos and mapa_setores:
            ticker_to_idx = {ticker: i for i, ticker in enumerate(nomes_ativos)}
            for setor in setores_proibidos:
                if setor in mapa_setores:
                    for ativo in mapa_setores[setor]:
                        # Remove espaços em branco extras para garantir o match
                        ativo_limpo = ativo.strip()
                        # Procura o ativo na lista de nomes (também limpa)
                        if ativo_limpo in ticker_to_idx:
                            idx = ticker_to_idx[ativo_limpo]
                            xu[idx] = 0.0
                            
        # Garante que não ficou nada negativo
        xu = np.maximum(0.0, xu)

        # Inicializa o problema
        super().__init__(n_var=n_ativos, n_obj=1, n_constr=2, n_eq_constr=0, xl=xl, xu=xu)

    # Função de Avaliação
    def _evaluate(self, x, out, *args, **kwargs):

        # 1. Retorno Esperado
        retorno_port = x.dot(self.retornos_medios)
        
        # 2. Variância (Risco)
        variancia = np.einsum('...i,ij,...j->...', x, self.matriz_cov, x)
        risco_vol = np.sqrt(np.maximum(variancia, 1e-12))
        
        # 3. P/VP e CVaR 
        pvp_port = x.dot(self.vetor_pvp)
        cvar_port = x.dot(self.vetor_cvar)
        
        # 4. Penalidade por Caixa Não Investido
        soma_pesos = np.sum(x, axis=1)
        caixa_nao_investido = np.maximum(0.0, 1.0 - soma_pesos)
        penalidade_caixa = config.PESO_PENALIZACAO_CAIXA * caixa_nao_investido

        # Função Objetivo (Multiobjetivo Scalarizado)
        # Minimizamos: (Risco * Lambda) - Retorno + Custo P/VP + Custo CVaR + Penalidade Caixa
        obj = (self.lambda_aversao_risco * variancia) - retorno_port \
              + (config.PESO_PVP * pvp_port) \
              + (config.PESO_CVAR * cvar_port) \
              + penalidade_caixa
        
        out["F"] = obj
        
        # Restrições

        # 1. Risco <= Risco Máximo 
        g1 = risco_vol - self.risco_maximo_usuario
        
        # 2. Soma dos Pesos <= 1.0 
        g2 = np.sum(x, axis=1) - 1.0
        
        # Empilha as restrições na saída "G"
        out["G"] = np.column_stack([g1, g2])




# Função Repair personalizada para impor tetos setoriais
class SectorCapRepair(Repair):
    def __init__(self, mapa_setores, nomes_ativos, teto_setor, xu):
       
        super().__init__() 
        
        # Limite máximo por setor (ex: 0.10 para 10%)
        self.teto_setor = teto_setor
        self.xu = xu 

        self.indices_setores = [] 
        if mapa_setores:
            for setor, lista_ativos_setor in mapa_setores.items():
                idxs = [i for i, nome in enumerate(nomes_ativos) if nome in lista_ativos_setor]
                if idxs:
                    self.indices_setores.append(idxs)

    def _do(self, problem, X, **kwargs):

        # 1. Trata NaNs/Infs
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 2. Restrição de Mínimo de Peso (0.5%)
        X[X < 0.005] = 0.0
        
        # 3. Restrição de Teto Individual (xu)
        if self.xu is not None:
             X = np.minimum(X, self.xu)

        # 4. Garantia de Orçamento (Soma ≤ 1.0)
        somas = X.sum(axis=1, keepdims=True)

        mask_estouro_orcamento = somas > 1.0
        fatores = np.ones_like(somas)
        np.putmask(fatores, mask_estouro_orcamento, 1.0 / (somas + 1e-9))
        X = X * fatores

        # 5. Aplicação do Teto Setorial
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
        
        # 6. Reaplica o mínimo de peso (0.5%) após ajustes
        X[X < 0.005] = 0.0

        return X

# Função principal para rodar a otimização via AG
def rodar_otimização(inputs, risco_maximo_usuario, lambda_aversao_risco, 
                     setores_proibidos=None, 
                     teto_maximo_ativo=0.30, 
                     teto_maximo_setor=1.0,
                     verbose=True):
    
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

    if verbose:
        print("\n[GA] Inicializando Modelo Multiobjetivo...")
    
    # 2. Instancia o Problema
    problema = OtimizacaoPortfolio(
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
        teto_maximo_setor=teto_maximo_setor,
        verbose=verbose  
    )
    
    if verbose:
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
        if verbose:
            print(f"Otimização concluída após {res.algorithm.n_gen} gerações.")
            print()
        pesos_otimos = res.X
        utility_score = res.F[0]
        
        # Verifica se houve "Caixa" (Soma < 1.0)
        soma_pesos = np.sum(pesos_otimos)
        if soma_pesos < 0.99 and verbose:
            print(f"[GA] Aviso: Restrições impediram 100% de alocação. Investido: {soma_pesos:.1%}")
        
        # Métricas Finais
        variancia = pesos_otimos.dot(matriz_cov).dot(pesos_otimos)
        risco_otimo = np.sqrt(variancia)
        retorno_otimo = pesos_otimos.dot(retornos_medios)
        pvp_final = pesos_otimos.dot(vetor_pvp)
        cvar_final = pesos_otimos.dot(vetor_cvar)
        
        # Recalcula a função objetivo padrão para comparação justa com Gurobi
        penalidade_caixa = config.PESO_PENALIZACAO_CAIXA * max(0.0, 1.0 - pesos_otimos.sum())
        
        utility_score_reporting = (lambda_aversao_risco * (risco_otimo ** 2)) - retorno_otimo \
                                  + (config.PESO_PVP * pvp_final) \
                                  + (config.PESO_CVAR * cvar_final) \
                                  + penalidade_caixa
        
        # DataFrame para retorno (usado pelo app.py)
        df_pesos = pd.DataFrame([pesos_otimos], columns=nomes_dos_ativos)
        df_objetivos = pd.DataFrame({
            'Risco_Alvo': [risco_maximo_usuario],
            'Risco_Encontrado_Anual': [risco_otimo],
            'Retorno_Encontrado_Anual': [retorno_otimo]
        })
        df_final = pd.concat([df_objetivos, df_pesos], axis=1)
        
        # 6. Retorno dos Resultados
        return {
            "metricas": {
                "retorno_aa": retorno_otimo * 100,
                "risco_aa": risco_otimo * 100,
                "score": utility_score_reporting,
                "funcao_objetivo": utility_score,
                "lambda_risco": lambda_aversao_risco,
                "pvp_final": pvp_final,   
                "cvar_final": cvar_final
            },
            "dataframe_resultado": df_final,
            "pesos_finais": pesos_otimos,
            "risco_final": risco_otimo,
            "retorno_final": retorno_otimo,
            "funcao_objetivo": utility_score
        }
            
    else:
        if verbose:
            print("\nALERTA: Otimização não convergiu para uma solução viável.")
        return None

