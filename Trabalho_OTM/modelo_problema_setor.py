import numpy as np
from pymoo.core.problem import Problem
import config 

class OtimizacaoPortfolio(Problem):
    def __init__(self, retornos_medios, matriz_cov, 
                 vetor_pvp, vetor_cvar, 
                 volume_medio, valor_investido, # <--- NOVOS ARGS
                 risco_maximo_usuario, lambda_aversao_risco,
                 nomes_ativos=None, mapa_setores=None, setores_proibidos=None):
        
        self.retornos_medios = retornos_medios
        self.matriz_cov = matriz_cov
        self.vetor_pvp = vetor_pvp.values     
        self.vetor_cvar = vetor_cvar.values   
        
        self.risco_maximo_usuario = risco_maximo_usuario
        self.lambda_aversao_risco = lambda_aversao_risco
        
        n_ativos = len(retornos_medios)
        
        xl = np.full(n_ativos, 0.0)
        
        # --- CÁLCULO DOS LIMITES SUPERIORES (LIQUIDEZ) ---
        # Regra: Alocação (R$) <= 1% do Volume Médio Diário
        # Alocação (R$) = Peso * Valor_Investido
        # Logo: Peso <= (0.01 * Volume) / Valor_Investido
        
        # 1. Calcula o teto financeiro permitido por ativo
        teto_financeiro_liquidez = 0.1 * volume_medio.values
        
        # 2. Converte para teto em peso (0 a 1)
        # Evita divisão por zero se valor investido for nulo (apenas segurança)
        inv = valor_investido if valor_investido > 0 else 1.0
        max_peso_liquidez = teto_financeiro_liquidez / inv
        
        # 3. O limite final é o menor entre: 1.0 (100%) e o Teto de Liquidez
        # Ex: Se liquidez permite 200% da carteira, trava em 100%.
        # Ex: Se liquidez permite apenas 5%, trava em 5%.
        xu = np.minimum(0.300, max_peso_liquidez)
        
        # --- CÁLCULO DOS LIMITES (SETORES PROIBIDOS) ---
        if setores_proibidos and nomes_ativos and mapa_setores:
            ticker_to_idx = {ticker: i for i, ticker in enumerate(nomes_ativos)}
            for setor in setores_proibidos:
                if setor in mapa_setores:
                    for ativo in mapa_setores[setor]:
                        if ativo in ticker_to_idx:
                            idx = ticker_to_idx[ativo]
                            xu[idx] = 0.0 # Força zero (sobrescreve liquidez)
                            
        # Verifica se algum limite ficou negativo (erro de dados) e corrige pra 0
        xu = np.maximum(0.0, xu)
        
        # Debug: Mostrar quantos ativos foram limitados pela liquidez
        # Consideramos limitado se o teto for menor que 0.10 (10%)
        limitados = np.sum((xu < 0.1) & (xu > 0.0))
        print(f"--> Restrição de Liquidez: {limitados} ativos limitados a <10% da carteira.")
                            
        super().__init__(n_var=n_ativos, n_obj=1, n_constr=1, n_eq_constr=1, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        # 1. Retorno Esperado
        retorno_port = x.dot(self.retornos_medios)
        
        # 2. Variância
        variancia = np.einsum('...i,ij,...j->...', x, self.matriz_cov, x)
        risco_vol = np.sqrt(np.maximum(variancia, 1e-12))
        
        # 3. P/VP e CVaR
        pvp_port = x.dot(self.vetor_pvp)
        cvar_port = x.dot(self.vetor_cvar)
        
        # Função Objetivo (Multiobjetivo Scalarizado)
        # Pesos calibrados no config.py
        obj = (self.lambda_aversao_risco * variancia) - retorno_port \
              + (config.PESO_PVP * pvp_port) \
              + (config.PESO_CVAR * cvar_port)
        
        out["F"] = obj
        
        # Restrições
        out["H"] = np.sum(x, axis=1) - 1.0
        out["G"] = risco_vol - self.risco_maximo_usuario