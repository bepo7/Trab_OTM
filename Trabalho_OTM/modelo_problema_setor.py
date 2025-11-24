import numpy as np
from pymoo.core.problem import Problem

class OtimizacaoPortfolioSetor(Problem):
    """
    Modelo OTIMIZADO para restrição de setores com Função Objetivo Média-Variância.
    
    Objetivo:
    Minimizar (Lambda * Variância - Retorno)
    
    Usa 'Box Constraints' (limites superiores = 0) para os ativos proibidos.
    Isso garante que o algoritmo genético nunca perca tempo testando
    pesos > 0 para setores que o usuário não quer.
    """

    def __init__(self, retornos_medios, matriz_cov, 
                 risco_maximo_usuario, lambda_aversao_risco,
                 nomes_ativos_ordenados, mapa_setores, setores_proibidos):
        
        self.retornos_medios = retornos_medios
        self.matriz_cov = matriz_cov
        self.risco_maximo_usuario = risco_maximo_usuario
        self.lambda_aversao_risco = lambda_aversao_risco
        
        n_ativos = len(retornos_medios)
        
        # --- 1. Definir Limites Padrão (0% a 100%) ---
        xl = np.full(n_ativos, 0.0) # Limite Inferior
        xu = np.full(n_ativos, 1.0) # Limite Superior (padrão é 1.0)

        # --- 2. Aplicar Restrição de Setor (Forçar Limite Superior a 0%) ---
        ticker_to_idx = {ticker: i for i, ticker in enumerate(nomes_ativos_ordenados)}
        ativos_zerados_count = 0
        
        for setor in setores_proibidos:
            if setor in mapa_setores:
                for ativo in mapa_setores[setor]:
                    if ativo in ticker_to_idx:
                        idx = ticker_to_idx[ativo]
                        # AQUI ESTÁ A MÁGICA: Força o peso máximo a ser ZERO
                        xu[idx] = 0.0
                        ativos_zerados_count += 1

        print(f"--> Modelo Setorial: {ativos_zerados_count} ativos tiveram seus pesos travados em 0.")

        # --- 3. Inicializar Problema ---
        # Note que voltamos a ter apenas 1 restrição de desigualdade (o Risco)
        # porque as restrições de setor agora estão embutidas no 'xu'
        super().__init__(
            n_var=n_ativos,
            n_obj=1,
            n_constr=1,    # Apenas 1: Teto de Risco (Desvio Padrão)
            n_eq_constr=1, # Apenas 1: Soma pesos = 1
            xl=xl,
            xu=xu          # Passamos os novos limites superiores
        )

    def _evaluate(self, x, out, *args, **kwargs):
        # --- Cálculos Padrão ---
        
        # Retorno: (w * mu)
        retorno_calculado = x.dot(self.retornos_medios)
        
        # Variância: (w * COV * w.T)
        variancia = np.einsum('...i,ij,...j->...', x, self.matriz_cov, x)
        
        # Desvio Padrão (Risco): sqrt(Variância)
        # Calculamos isso APENAS para a restrição do usuário
        risco_calculado = np.sqrt(np.maximum(variancia, 1e-12))
        
        # --- OBJETIVO (ALTERADO PARA VARIÂNCIA) ---
        # Antes: (Lambda * Risco_Desvio_Padrao) - Retorno
        # Agora: (Lambda * Variancia) - Retorno
        out["F"] = (self.lambda_aversao_risco * variancia) - retorno_calculado
        
        # --- RESTRIÇÕES ---
        
        # Restrição de Igualdade (Soma = 1)
        out["H"] = np.sum(x, axis=1) - 1.0
        
        # Restrição de Desigualdade (Risco Teto <= Max)
        # Mantemos em Desvio Padrão para ser consistente com a entrada do usuário (ex: 15%)
        out["G"] = risco_calculado - self.risco_maximo_usuario