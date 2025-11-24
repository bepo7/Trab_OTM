import numpy as np
from pymoo.core.problem import Problem

class OtimizacaoPortfolio(Problem):
    """
    Esta classe define o problema de Otimização de Portfólio
    (Média-Variância) atualizado para usar VARIÂNCIA na função objetivo.
    
    Objetivo:
    Maximizar (Retorno - Lambda * Variância)
    => Minimizar (Lambda * Variância - Retorno)
    
    Sujeito a:
    1. Risco (Desvio Padrão) <= Risco_Maximo_Usuario (Restrição de Teto mantida na escala original)
    2. Soma(w) = 1
    """

    def __init__(self, retornos_medios, matriz_cov, 
                 risco_maximo_usuario, lambda_aversao_risco):
        """
        Inicializa o problema de otimização.
        """
        
        # --- Parâmetros de Entrada ---
        self.retornos_medios = retornos_medios
        self.matriz_cov = matriz_cov
        self.risco_maximo_usuario = risco_maximo_usuario
        self.lambda_aversao_risco = lambda_aversao_risco 
        
        n_ativos = len(retornos_medios)
        
        # --- Definição Formal do Problema para o Pymoo ---
        super().__init__(
            n_var=n_ativos,       
            n_obj=1,              # Objetivo Único
            n_constr=1,           # Restrição de Desigualdade (Risco Teto)
            n_eq_constr=1,        # Restrição de Igualdade (Soma = 1)
            xl=np.full(n_ativos, 0.0), # Lower bound
            xu=np.full(n_ativos, 1.0)  # Upper bound
        )

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Calcula os objetivos e restrições para um conjunto 'x' de portfólios.
        """
        
        # --- 1. CÁLCULOS BÁSICOS ---
        
        # Retorno: (w * mu)
        retorno_calculado = x.dot(self.retornos_medios)
        
        # Variância: (w * COV * w.T)
        # Nota: Usamos einsum para cálculo eficiente em lote (batch)
        variancia = np.einsum('...i,ij,...j->...', x, self.matriz_cov, x)
        
        # Desvio Padrão (Risco): sqrt(Variância)
        # Mantemos este cálculo para usar na RESTRIÇÃO de teto do usuário
        risco_calculado = np.sqrt(np.maximum(variancia, 1e-12)) 
        
        
        # --- 2. OBJETIVO (ALTERADO PARA VARIÂNCIA) ---
        
        # Julia (Mosek): Min( -Retorno + Lambda * Variancia )
        # Python (Pymoo): Min( Lambda * Variancia - Retorno )
        # ---------------------------------------------------------
        # MUDANÇA AQUI: Trocamos 'risco_calculado' por 'variancia'
        # ---------------------------------------------------------
        funcao_objetivo = (self.lambda_aversao_risco * variancia) - retorno_calculado
        
        
        # --- 3. RESTRIÇÕES ---
        
        # IGUALDADE: Soma dos Pesos - 1 = 0
        restricao_soma_pesos = np.sum(x, axis=1) - 1.0
        
        # DESIGUALDADE: Risco da Carteira - Risco_Max <= 0
        # (Mantemos em Desvio Padrão para ser fiel ao input do usuário ex: 0.15)
        restricao_de_risco_teto = risco_calculado - self.risco_maximo_usuario

        # --- 4. Envio dos Resultados ---
        
        out["F"] = funcao_objetivo
        out["G"] = restricao_de_risco_teto
        out["H"] = restricao_soma_pesos