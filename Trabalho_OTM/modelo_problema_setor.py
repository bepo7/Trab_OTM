import numpy as np
from pymoo.core.problem import Problem

class OtimizacaoPortfolio(Problem):
    """
    Modelo UNIFICADO de Otimização de Portfólio (Média-Variância).
    
    Funcionalidade:
    - Minimizar (Lambda * Variância - Retorno).
    - Suporta restrição de soma = 1.
    - Suporta teto de risco.
    - Suporta (Opcional) exclusão de setores via 'Box Constraints' (limite superior = 0).
    """

    def __init__(self, retornos_medios, matriz_cov, 
                 risco_maximo_usuario, lambda_aversao_risco,
                 nomes_ativos=None, mapa_setores=None, setores_proibidos=None):
        """
        Inicializa o problema.
        Se 'setores_proibidos' for fornecido, os ativos desses setores terão peso máximo travado em 0.0.
        Caso contrário, todos podem ir até 1.0 (100%).
        """
        
        self.retornos_medios = retornos_medios
        self.matriz_cov = matriz_cov
        self.risco_maximo_usuario = risco_maximo_usuario
        self.lambda_aversao_risco = lambda_aversao_risco
        
        n_ativos = len(retornos_medios)
        
        # --- 1. Definir Limites Padrão (0% a 100%) ---
        xl = np.full(n_ativos, 0.0) # Limite Inferior (Sempre 0)
        xu = np.full(n_ativos, 1.0) # Limite Superior (Padrão é 1.0)

        # --- 2. Aplicar Restrição de Setor (Se houver) ---
        # Se o usuário mandou travar setores, identificamos os ativos e mudamos o 'xu' para 0.0
        ativos_zerados_count = 0
        if setores_proibidos and nomes_ativos and mapa_setores:
            print(f"--> Aplicando restrições para setores: {setores_proibidos}")
            
            # Cria mapa reverso: Ticker -> Índice
            ticker_to_idx = {ticker: i for i, ticker in enumerate(nomes_ativos)}
            
            for setor in setores_proibidos:
                if setor in mapa_setores:
                    for ativo in mapa_setores[setor]:
                        if ativo in ticker_to_idx:
                            idx = ticker_to_idx[ativo]
                            xu[idx] = 0.0 # Trava o ativo
                            ativos_zerados_count += 1
            
            print(f"--> Total de ativos travados em 0.0: {ativos_zerados_count}")
        else:
            print("--> Otimização sem restrições de setor (Todos ativos liberados).")

        # --- 3. Inicializar Problema Pymoo ---
        super().__init__(
            n_var=n_ativos,
            n_obj=1,       # 1 Objetivo (Minimizar Função Utilidade)
            n_constr=1,    # 1 Restrição Desigualdade (Risco Teto)
            n_eq_constr=1, # 1 Restrição Igualdade (Soma = 1)
            xl=xl,
            xu=xu          # Passamos os limites (que podem ter zeros ou não)
        )

    def _evaluate(self, x, out, *args, **kwargs):
        # --- Cálculos Vetorizados ---
        
        # 1. Retorno Esperado: (pesos * retornos)
        retorno_calculado = x.dot(self.retornos_medios)
        
        # 2. Variância: (pesos * COV * pesos_transposto)
        # einsum é usado para calcular a forma quadrática para múltiplas soluções (população) de uma vez
        variancia = np.einsum('...i,ij,...j->...', x, self.matriz_cov, x)
        
        # 3. Desvio Padrão (Apenas para verificar a restrição do usuário)
        risco_calculado = np.sqrt(np.maximum(variancia, 1e-12))
        
        # --- FUNÇÃO OBJETIVO ---
        # Minimizamos: (Aversão * Variância) - Retorno
        out["F"] = (self.lambda_aversao_risco * variancia) - retorno_calculado
        
        # --- RESTRIÇÕES ---
        
        # H (Igualdade): Soma dos pesos - 1.0 = 0
        out["H"] = np.sum(x, axis=1) - 1.0
        
        # G (Desigualdade): Risco Calculado - Risco Teto <= 0
        out["G"] = risco_calculado - self.risco_maximo_usuario