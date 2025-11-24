import config
import preparar_dados
import otimizar
import pandas as pd
import plot
import os

def limpar_tela():
    os.system('cls' if os.name == 'nt' else 'clear')

def menu_selecao_setores():
    """
    Menu interativo onde o usuário MARCA os setores que deseja INCLUIR.
    Começa com tudo desmarcado (proibido).
    Retorna a lista final de setores PROIBIDOS (os que não foram marcados).
    """
    print("\n=== FILTRO DE SETORES ===")
    # Se o usuário NÃO quiser personalizar, assumimos que ele quer TUDO (comportamento padrão)
    resp = input("Deseja escolher manualmente os setores permitidos? (s/N): ").strip().lower()
    if resp != 's':
        return [] # Retorna lista vazia de proibidos -> TUDO PERMITIDO

    mapa = config.obter_mapa_setores_ativos()
    setores = list(mapa.keys())
    
    # MUDANÇA AQUI: Inicialmente tudo False (Proibido/Não selecionado)
    # O usuário vai marcar o que ele QUER.
    estado_setores = {s: False for s in setores}

    while True:
        limpar_tela()
        print("=== SELEÇÃO DE SETORES ===")
        print("Todos começam desativados. Digite o NÚMERO para ATIVAR os desejados.\n")
        print("STATUS ATUAL:")
        
        ativados_count = 0
        for i, setor in enumerate(setores):
            # [✓] se True (Ativo), [ ] se False (Inativo)
            status = "[✓] SELECIONADO" if estado_setores[setor] else "[ ] ignorado"
            if estado_setores[setor]:
                ativados_count += 1
                
            n_ativos = len(mapa[setor])
            print(f"{i+1:2d}. {status} - {setor:<35} ({n_ativos} ativos)")
            
        print(f"\nSetores selecionados: {ativados_count} de {len(setores)}")
        print("[C] CONCLUIR seleção e otimizar")
        print("[T] Selecionar TODOS")
        print("[N] Desmarcar TODOS")
        
        opcao = input("Opção: ").strip().lower()

        if opcao == 'c':
            if ativados_count == 0:
                print("\nAVISO: Você não selecionou nenhum setor!")
                conf = input("Deseja prosseguir mesmo assim (vai gerar erro)? (s/N): ")
                if conf.lower() == 's': break
            else:
                break
        elif opcao == 't':
            estado_setores = {s: True for s in setores}
        elif opcao == 'n':
            estado_setores = {s: False for s in setores}
        else:
            try:
                idx = int(opcao) - 1
                if 0 <= idx < len(setores):
                    set_selecionado = setores[idx]
                    # Inverte o estado atual
                    estado_setores[set_selecionado] = not estado_setores[set_selecionado]
            except ValueError:
                pass

    # A função deve retornar os PROIBIDOS para o modelo matemático.
    # Proibidos são aqueles que permaneceram como False.
    proibidos = [s for s, ativo in estado_setores.items() if not ativo]
    
    return proibidos

# --- O RESTANTE DO ARQUIVO CONTINUA IGUAL ---
# (imprimir_carteira_final e main não precisam de mudanças drásticas, 
# apenas garantindo que main chame essa nova versão do menu)

def imprimir_carteira_final(df_resultado, valor_investido):
    print("\n--- RESUMO DA CARTEIRA OTIMIZADA ---")
    try:
        row = df_resultado.iloc[0]
    except: return
    
    print(f"Risco (Volatilidade Anual): {row['Risco_Encontrado_Anual']:.2%}")
    print(f"Retorno Esperado (Anual):   {row['Retorno_Encontrado_Anual']:.2%}")
    print("\nTop Alocações:")
    cols_ignorar = ['Risco_Alvo', 'Risco_Encontrado_Anual', 'Retorno_Encontrado_Anual']
    pesos = row.drop(cols_ignorar, errors='ignore')
    pesos = pesos[pesos > 0.001].sort_values(ascending=False)
    
    for ticker, peso in pesos.head(15).items():
        val = peso * valor_investido
        print(f"{ticker:<12} | {peso:6.2%} | R$ {val:10,.2f}")
    print("-" * 40)

def main():
    # ... (mesmo código de inicialização da resposta anterior)
    limpar_tela()
    print("--- OTIMIZADOR V4 ---")
    try:
        valor = float(input("Valor Total (R$): "))
        risco_teto = float(input("Teto de Risco (%): ")) / 100.0
        l_in = input(f"Lambda [Enter={config.LAMBDA_AVERSAO_RISCO_PADRAO}]: ")
        lambda_risco = float(l_in) if l_in else config.LAMBDA_AVERSAO_RISCO_PADRAO
    except ValueError: return

    # Chama o novo menu
    setores_proibidos = menu_selecao_setores()

    print("\n[1/3] Preparando dados...")
    inputs = preparar_dados.calcular_inputs_otimizacao(valor)
    if not inputs: return

    print("\n[2/3] Otimizando...")
    res = otimizar.rodar_otimização(inputs, risco_teto, lambda_risco, setores_proibidos)
    if not res: return

    print("\n[3/3] Resultados...")
    imprimir_carteira_final(res['dataframe_resultado'], valor)
    try: plot.rodar_visualizacao(inputs, res)
    except: pass
    print("\nFIM.")

if __name__ == '__main__':
    main()