import time
import pandas as pd
import re
import math
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import subprocess
import psutil
import os
import random
import time
import re
import math
from scipy.spatial import distance_matrix
import time
import time
import pandas as pd  # Importar pandas para manipulação de DataFrames
import random
import numpy as np
import concurrent.futures  # Para implementar o timeout
import traceback  # Para capturar mensagens de erro

class AnalisePreviaEstatistica:
    pass

class PreProcessingInstancia:
    def __init__(self, arquivo, velocidade_maxima=1.0):
        # Inicia o contador de tempo
        self.inicio = time.time()
        
        # Armazena parâmetros
        self.arquivo = arquivo
        self.velocidade_maxima = velocidade_maxima
        
        # Parâmetros extraídos do arquivo
        self.num_cidades = int("".join(filter(str.isdigit, self.ler_linha_especifica(3))))
        self.n_item = int("".join(filter(str.isdigit, self.ler_linha_especifica(4))))
        self.capacidade_mochila = int("".join(filter(str.isdigit, self.ler_linha_especifica(5))))
        self.velocidade_minima = float(re.findall(r'\d+\.\d+', self.ler_linha_especifica(6))[0])
        self.taxa_aluguel_por_tempo = float(re.findall(r'\d+\.\d+', self.ler_linha_especifica(8))[0])
        
        # Ler o arquivo a partir da linha das cidades
        self.cabecalho, self.cidades, self.itens = self.ler_arquivo_a_partir_da_linha(self.num_cidades)
        
        # Criar DataFrame para cidades
        self.cidades = [linha.replace('NODE_COORD_SECTION\t(INDEX, X, Y): ', 'cidade\tx\ty') for linha in self.cidades]
        self.cidades_df = self.criar_tabela(self.cidades)
        
        # Criar DataFrame para itens
        itens = [linha.replace('ITEMS SECTION\t(INDEX, PROFIT, WEIGHT, ASSIGNED NODE NUMBER): ', 'item\tprofit\tweight\tcidade') for linha in self.itens]
        self.itens_df = self.criar_tabela(itens)
        # Estrutura para itens associados a cidades
        #self.itens = [(int(self.itens_df['cidade'].iloc[i]), int(self.itens_df['profit'].iloc[i]), int(self.itens_df['weight'].iloc[i])) for i in range(len(self.itens_df))]
        
        # Converter colunas para tipos numéricos
        for coluna in self.itens_df.columns:
            self.itens_df[coluna] = pd.to_numeric(self.itens_df[coluna], errors='coerce')
        self.itens_df = self.itens_df.dropna()  # Remove linhas com valores NaN, se houver
        
        # Estrutura para itens associados a cidades
        self.itens_associados = [
            (
                int(self.itens_df['cidade'].iloc[i]) - 1,  # Subtrai 1 para ajustar o índice da cidade
                int(self.itens_df['profit'].iloc[i]),
                int(self.itens_df['weight'].iloc[i])
            ) for i in range(len(self.itens_df))
        ]
        
        # Convertendo as coordenadas das cidades para arrays NumPy
        self.coordenadas = self.cidades_df[['x', 'y']].astype(float).values
        
        # Calculando a matriz de distâncias com scipy
        self.distancia_cidades = distance_matrix(self.coordenadas, self.coordenadas)
        
        # Coordenadas das cidades para plotagem
        self.coordenadas_cidades = {
            i: (float(self.cidades_df['x'].iloc[i]), float(self.cidades_df['y'].iloc[i])) for i in range(len(self.cidades_df))
        }
    
    # Função para ler o arquivo a partir de uma linha específica
    def ler_arquivo_a_partir_da_linha(self, linha_inicial):
        with open(self.arquivo, 'r') as file:
            linhas = file.readlines()
        cabecalho = linhas[:8]  # primeiras 8 linhas para cabeçalho
        cidades = linhas[9:linha_inicial + 10]  # linhas contendo as cidades
        itens = linhas[linha_inicial + 10:]  # linhas contendo os itens
        return cabecalho, cidades, itens
    
    # Função para ler uma linha específica do arquivo
    def ler_linha_especifica(self, numero_linha):
        with open(self.arquivo, 'r') as file:
            linhas = file.readlines()
        if 1 <= numero_linha <= len(linhas):
            return linhas[numero_linha - 1]
    
    # Função para criar um DataFrame a partir das linhas lidas
    def criar_tabela(self, linhas):
        # Supondo que os dados estejam separados por tabulações
        dados = [linha.strip().split('\t') for linha in linhas]
        df = pd.DataFrame(dados[1:], columns=dados[0])  # Define a primeira linha como cabeçalho
        
        # Converter todas as colunas para tipos numéricos
        for coluna in df.columns:
            df[coluna] = pd.to_numeric(df[coluna], errors='coerce')
        df = df.dropna()  # Remove linhas com valores NaN, se houver
        return df

class TSPSolver:
    def __init__(self, matriz_distancias, caminho_executavel, nome_problema="exemplo_tsp"):
        self.matriz_distancias = matriz_distancias
        self.caminho_executavel = caminho_executavel
        self.nome_problema = nome_problema
        self.nome_arquivo_tsp = f"{nome_problema}.tsp"
        self.arquivo_parametros = f"{nome_problema}.par"
        self.arquivo_saida = f"{nome_problema}.tour"
        self.rota_otimizada = None
        self.distancia_total = None

    # Função para escrever o arquivo .tsp no formato TSPLIB
    def escrever_arquivo_tsp(self):
        """
        Escreve um arquivo .tsp no formato TSPLIB usando a matriz de distâncias fornecida.
        """
        matriz_distancias = self.matriz_distancias

        # Verifica se a matriz de distâncias é quadrada
        if matriz_distancias.shape[0] != matriz_distancias.shape[1]:
            raise ValueError("A matriz de distâncias deve ser quadrada (mesmo número de linhas e colunas).")

        n = matriz_distancias.shape[0]  # Número de cidades

        with open(self.nome_arquivo_tsp, 'w') as file:
            # Escreve o cabeçalho do arquivo .tsp
            file.write(f"NAME : {self.nome_problema}\n")
            file.write("TYPE : TSP\n")
            file.write(f"DIMENSION : {n}\n")
            file.write("EDGE_WEIGHT_TYPE : EXPLICIT\n")
            file.write("EDGE_WEIGHT_FORMAT : FULL_MATRIX\n")
            file.write("EDGE_WEIGHT_SECTION\n")
            
            # Escreve a matriz de distâncias
            for i in range(n):
                linha = ' '.join(map(str, matriz_distancias[i]))
                file.write(f"{linha}\n")
            
            # Indica o fim do arquivo
            file.write("EOF\n")
        print(f"Arquivo {self.nome_arquivo_tsp} criado com sucesso.")

    # Função para criar o arquivo de parâmetros para o LKH-3
    def escrever_arquivo_parametros(self):
        """
        Escreve o arquivo de parâmetros necessário para o LKH-3.
        """
        with open(self.arquivo_parametros, 'w') as file:
            file.write(f"PROBLEM_FILE = {self.nome_arquivo_tsp}\n")
            file.write(f"OUTPUT_TOUR_FILE = {self.arquivo_saida}\n")
        print(f"Arquivo de parâmetros {self.arquivo_parametros} criado com sucesso.")

    # Função para executar o LKH-3 até gerar o arquivo de saída
    def executar_lkh3(self, timeout=120):
        """
        Executa o LKH-3 até que o arquivo de saída seja gerado ou o timeout total seja atingido.
        """
        caminho_executavel = self.caminho_executavel
        caminho_parametros = self.arquivo_parametros
        arquivo_saida = self.arquivo_saida

        # Verifica se o executável e o arquivo de parâmetros existem
        if not os.path.isfile(caminho_executavel):
            print(f"Erro: O executável {caminho_executavel} não foi encontrado.")
            return None
        if not os.path.isfile(caminho_parametros):
            print(f"Erro: O arquivo de parâmetros {caminho_parametros} não foi encontrado.")
            return None

        # Exclui o arquivo de saída, se ele já existir
        if os.path.isfile(arquivo_saida):
            os.remove(arquivo_saida)
            print(f"Arquivo de saída '{arquivo_saida}' existente foi excluído.")

        try:
            # Inicia o subprocesso
            processo = subprocess.Popen(
                [caminho_executavel, caminho_parametros],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Monitora a criação do arquivo de saída ou até o timeout total ser atingido
            inicio = time.time()
            while processo.poll() is None:
                # Verifica se o arquivo de saída foi gerado
                if os.path.isfile(arquivo_saida):
                    print(f"Arquivo de saída '{arquivo_saida}' encontrado. Encerrando o processo.")
                    processo.terminate()
                    time.sleep(1)

                    # Força o encerramento de subprocessos, se necessário
                    processo_psutil = psutil.Process(processo.pid)
                    for proc in processo_psutil.children(recursive=True):
                        proc.terminate()
                    processo_psutil.terminate()

                    # Verificação final de processos que ainda estão ativos
                    gone, still_alive = psutil.wait_procs([processo_psutil], timeout=1)
                    if still_alive:
                        for proc in still_alive:
                            proc.kill()  # Mata qualquer processo que ainda esteja ativo

                    if processo.poll() is None:
                        processo.kill()  # Encerra o processo principal, se ainda estiver ativo
                    break

                # Verifica se o tempo limite total foi atingido
                if time.time() - inicio > timeout:
                    print("Tempo limite total atingido. Encerrando o processo.")
                    processo.terminate()
                    time.sleep(1)
                    processo_psutil = psutil.Process(processo.pid)
                    for proc in processo_psutil.children(recursive=True):
                        proc.terminate()
                    processo_psutil.terminate()
                    gone, still_alive = psutil.wait_procs([processo_psutil], timeout=1)
                    if still_alive:
                        for proc in still_alive:
                            proc.kill()
                    if processo.poll() is None:
                        processo.kill()
                    break

                time.sleep(0.5)  # Aguarda brevemente antes de verificar novamente

            # Captura a saída e os erros do processo, caso ele tenha terminado
            stdout, stderr = processo.communicate()
            if processo.returncode == 0:
                print("LKH-3 finalizado com sucesso.")
                return stdout
            else:
                print("Erro ao executar o LKH-3:", stderr)
                return None
        except Exception as e:
            print("Erro durante a execução:", e)
            return None

    # Função para ler a rota otimizada do arquivo de saída
    def ler_rota_otimizada(self):
        """
        Lê a rota otimizada a partir do arquivo de saída .tour do LKH-3.
        """
        arquivo_saida = self.arquivo_saida
        rota = []
        try:
            with open(arquivo_saida, 'r') as file:
                leitura = False
                for linha in file:
                    # Inicia a leitura ao encontrar "TOUR_SECTION"
                    if "TOUR_SECTION" in linha:
                        leitura = True
                        continue
                    # Encerra a leitura ao encontrar o marcador "-1"
                    if leitura:
                        if linha.strip() == "-1":
                            break
                        # Adiciona o índice da cidade à rota
                        rota.append(int(linha.strip()) - 1)  # Subtrai 1 para índices zero-based
            self.rota_otimizada = rota
            print("Rota otimizada lida com sucesso.")
            return rota
        except FileNotFoundError:
            print(f"Erro: o arquivo {arquivo_saida} não foi encontrado.")
            return None
        except Exception as e:
            print(f"Ocorreu um erro ao ler o arquivo: {e}")
            return None

    # Função para calcular a distância total da rota otimizada
    def calcular_distancia_total(self):
        """
        Calcula a distância total da rota otimizada.
        """
        if self.rota_otimizada is None:
            print("Rota otimizada não foi lida ainda.")
            return None

        rota = self.rota_otimizada
        distancia_total = 0
        for i in range(len(rota) - 1):
            distancia_total += self.matriz_distancias[rota[i]][rota[i + 1]]
        distancia_total += self.matriz_distancias[rota[-1]][rota[0]]  # Retorno ao ponto inicial
        self.distancia_total = distancia_total
        print(f"Distância total calculada: {distancia_total}")
        return distancia_total

    # Método principal para executar todas as etapas
    def resolver(self, timeout=120):
        """
        Executa todas as etapas para resolver o TSP usando o LKH-3.
        """
        self.escrever_arquivo_tsp()
        self.escrever_arquivo_parametros()
        self.executar_lkh3(timeout=timeout)
        self.ler_rota_otimizada()
        self.calcular_distancia_total()

class ItemSelector:
    def __init__(self, capacidade_mochila):
        self.capacidade_mochila = capacidade_mochila

    def selecionar_itens(self, rota, itens):
        """
        Seleciona itens começando do fim da rota para o início,
        adicionando os itens de maior valor até atingir a capacidade da mochila.
        Retorna a lista de itens selecionados.
        """
        itens_selecionados = []
        peso_total = 0

        # Mapeamento de itens por cidade, ordenados pelo valor decrescente
        itens_por_cidade = {}
        for item in itens:
            cidade = item[0]
            if cidade not in itens_por_cidade:
                itens_por_cidade[cidade] = []
            itens_por_cidade[cidade].append(item)

        # Ordena os itens em cada cidade pelo valor decrescente
        for cidade in itens_por_cidade:
            itens_por_cidade[cidade].sort(key=lambda x: x[1], reverse=True)

        # Itera sobre a rota do fim para o início
        for cidade in reversed(rota):
            if cidade in itens_por_cidade:
                for item in itens_por_cidade[cidade]:
                    if peso_total + item[2] <= self.capacidade_mochila:
                        itens_selecionados.append(item)
                        peso_total += item[2]
                    if peso_total >= self.capacidade_mochila:
                        break  # Atingiu a capacidade máxima
            if peso_total >= self.capacidade_mochila:
                break  # Atingiu a capacidade máxima

        return itens_selecionados

class FuncaoObjetivo:
    def __init__(self, capacidade_mochila, taxa_aluguel_por_tempo, velocidade_maxima, velocidade_minima, distancia_cidades):
        self.capacidade_mochila = capacidade_mochila
        self.taxa_aluguel_por_tempo = taxa_aluguel_por_tempo
        self.velocidade_maxima = velocidade_maxima
        self.velocidade_minima = velocidade_minima
        self.distancia_cidades = distancia_cidades

    def calcular_velocidade(self, peso):
        velocidade = self.velocidade_maxima - (peso / self.capacidade_mochila) * (self.velocidade_maxima - self.velocidade_minima)
        return max(velocidade, self.velocidade_minima)  # Garante que a velocidade não seja menor que Vmin

    def calcular_lucro_com_itens(self, rota, itens_selecionados):
        peso_total = 0
        valor_total = sum(item[1] for item in itens_selecionados)
        tempo_total = 0
        aluguel_total = 0

        # Mapeamento de itens por cidade
        itens_por_cidade = {}
        for item in itens_selecionados:
            cidade = item[0]
            if cidade not in itens_por_cidade:
                itens_por_cidade[cidade] = []
            itens_por_cidade[cidade].append(item)

        for i in range(len(rota)):
            cidade_atual = rota[i]
            cidade_proxima = rota[i + 1] if i < len(rota) - 1 else rota[0]  # Após a última cidade, retorna ao início

            # Coleta de itens na cidade atual
            if cidade_atual in itens_por_cidade:
                for item in itens_por_cidade[cidade_atual]:
                    peso_total += item[2]
                    if peso_total > self.capacidade_mochila:
                        # Excede a capacidade; solução inválida
                        return 0, 0, 0, 0, float('-inf')

            # Calcula o tempo e aluguel para a cidade próxima
            distancia = self.distancia_cidades[cidade_atual][cidade_proxima]
            velocidade_atual = self.calcular_velocidade(peso_total)
            tempo = distancia / velocidade_atual
            tempo_total += tempo
            aluguel_total += self.taxa_aluguel_por_tempo * tempo

        lucro_viagem = valor_total - aluguel_total

        return valor_total, peso_total, tempo_total, aluguel_total, lucro_viagem

class BuscaTabuRotaItens:
    def __init__(self, capacidade_mochila, duracao_tabu, funcao_objetivo):
        self.capacidade_mochila = capacidade_mochila
        self.duracao_tabu = duracao_tabu
        self.funcao_objetivo = funcao_objetivo
        self.itens_indices = {}
        self.cidades_indices = {}
        self.tabu_lista_itens = {}
        self.tabu_lista_rotas = {}

    def inicializar_tabu_lists(self, itens, rota):
        # Inicializa as listas tabu para itens e rotas
        self.tabu_lista_itens = {}
        self.tabu_lista_rotas = {}
        # Mapeamento de índices para cidades
        for idx, cidade in enumerate(rota):
            self.cidades_indices[cidade] = idx
        # Mapeamento de índices para itens
        for idx, item in enumerate(itens):
            self.itens_indices[item] = idx

    def atualizar_tabu_lists(self):
        # Atualiza a lista tabu de itens
        itens_para_remover = []
        for movimento in self.tabu_lista_itens:
            self.tabu_lista_itens[movimento] -= 1
            if self.tabu_lista_itens[movimento] <= 0:
                itens_para_remover.append(movimento)
        for movimento in itens_para_remover:
            del self.tabu_lista_itens[movimento]

        # Atualiza a lista tabu de rotas
        rotas_para_remover = []
        for movimento in self.tabu_lista_rotas:
            self.tabu_lista_rotas[movimento] -= 1
            if self.tabu_lista_rotas[movimento] <= 0:
                rotas_para_remover.append(movimento)
        for movimento in rotas_para_remover:
            del self.tabu_lista_rotas[movimento]

    def gerar_vizinhos_rota_swap(self, rota_atual):
        vizinhos = []
        n = len(rota_atual)
        num_movimentos = min(100, (n - 1) * (n - 2) // 2)  # Limita a 10 movimentos para eficiência
        for _ in range(num_movimentos):
            i, j = random.sample(range(1, n - 1), 2)
            nova_rota = rota_atual[:]
            # Troca as posições das cidades i e j
            nova_rota[i], nova_rota[j] = nova_rota[j], nova_rota[i]
            movimento = (rota_atual[i], rota_atual[j])
            vizinhos.append((nova_rota, movimento))
        return vizinhos
    
    def gerar_vizinhos_rota_restrito_final_proporcionalmente(self, rota_atual):
        vizinhos = []
        n = len(rota_atual)
        n_corte = int(n*0.3)
        # Limitar aos movimentos 2-opt em uma amostra aleatória para eficiência
        num_movimentos = min(100, (n_corte - 1) * (n_corte - 2) // 2)  # Limita a 10 movimentos
        for _ in range(num_movimentos):
            i = random.randint(n_corte, n - 3)
            j = random.randint(i + 1, n - 2)
            # Realiza o movimento 2-opt
            nova_rota = rota_atual[:]
            nova_rota[i:j + 1] = reversed(nova_rota[i:j + 1])
            movimento = (rota_atual[i - 1], rota_atual[i], rota_atual[j], rota_atual[(j + 1) % n])
            vizinhos.append((nova_rota, movimento))
        return vizinhos
    
    def gerar_vizinhos_rota_cortes_limitados(self, rota_atual):
        vizinhos = []
        n = len(rota_atual)
        # Limitar aos movimentos 2-opt em uma amostra aleatória para eficiência
        num_movimentos = min(100, (n - 1) * (n - 2) // 2)  # Limita a 30 movimentos
        for _ in range(num_movimentos):
            i = random.randint(1, n - 3)
            j = random.randint(i + 1, n - 2) # Corte limitado em 10 cidades
            # Realiza o movimento 2-opt
            nova_rota = rota_atual[:]
            nova_rota[i:j + 1] = reversed(nova_rota[i:j + 1])
            movimento = (rota_atual[i - 1], rota_atual[i], rota_atual[j], rota_atual[(j + 1) % n])
            vizinhos.append((nova_rota, movimento))
        return vizinhos
    
    def gerar_vizinhos_rota(self, rota_atual):
        vizinhos = []
        n = len(rota_atual)
        # Limitar aos movimentos 2-opt em uma amostra aleatória para eficiência
        num_movimentos = min(10, (n - 1) * (n - 2) // 2)  # Limita a 15 movimentos
        for _ in range(num_movimentos):
            i = random.randint(1, n - 3)
            j = random.randint(i + 1, n - 2)
            # Realiza o movimento 2-opt
            nova_rota = rota_atual[:]
            nova_rota[i:j + 1] = reversed(nova_rota[i:j + 1])
            movimento = (rota_atual[i - 1], rota_atual[i], rota_atual[j], rota_atual[(j + 1) % n])
            vizinhos.append((nova_rota, movimento))
        return vizinhos

    def otimizar_mochila_e_rota_tabu(self, rota_inicial, itens_selecionados, itens, num_iteracoes):
        """
        Aplica Busca Tabu para otimizar a seleção de itens e a rota,
        garantindo que todas as soluções respeitem as restrições.
        """
        # Inicializa as estruturas necessárias
        self.inicializar_tabu_lists(itens, rota_inicial)

        # Solução atual
        solucao_atual_itens = itens_selecionados[:]
        rota_atual = rota_inicial[:]
        valor_total, peso_total, tempo_total, aluguel_total, lucro_atual = self.funcao_objetivo.calcular_lucro_com_itens(
            rota_atual, solucao_atual_itens
        )

        # Verificar se a solução inicial respeita a capacidade
        peso_atual = sum(item[2] for item in solucao_atual_itens)
        if peso_atual > self.capacidade_mochila:
            print("Solução inicial excede a capacidade da mochila.")
            return [], [], float('-inf')

        # Melhor solução encontrada
        melhor_itens = solucao_atual_itens[:]
        melhor_rota = rota_atual[:]
        melhor_lucro = lucro_atual

        for iteracao in range(num_iteracoes):
            vizinhos = []

            # Gera vizinhos de itens
            for item_incluido in solucao_atual_itens:
                for item_excluido in itens:
                    if item_excluido not in solucao_atual_itens:
                        # Se o item nao faz parte da solucao atual, seguimos:
                        movimento_itens = (item_incluido, item_excluido)
                        is_tabu = self.tabu_lista_itens.get(movimento_itens, 0) > 0

                        potencial_itens = solucao_atual_itens[:]
                        potencial_itens.remove(item_incluido)
                        potencial_itens.append(item_excluido)

                        # Verifica a restrição de capacidade
                        peso_potencial = sum(item[2] for item in potencial_itens)
                        if peso_potencial > self.capacidade_mochila:
                            continue  # Pula se violar a capacidade

                        # Calcula o lucro potencial
                        _, _, _, _, lucro_potencial = self.funcao_objetivo.calcular_lucro_com_itens(
                            rota_atual, potencial_itens
                        )

                        if not is_tabu or (is_tabu and lucro_potencial > melhor_lucro):
                            vizinhos.append({
                                "rota": rota_atual,
                                "itens": potencial_itens,
                                "lucro": lucro_potencial,
                                "movimento": movimento_itens,
                                "tipo_movimento": "itens"
                            })

            # Tenta adicionar um item
            for item in itens:
                if item not in solucao_atual_itens:
                    movimento_itens = ("add", item)
                    is_tabu = self.tabu_lista_itens.get(movimento_itens, 0) > 0

                    potencial_itens = solucao_atual_itens[:]
                    potencial_itens.append(item)

                    # Verifica a restrição de capacidade
                    peso_potencial = sum(item[2] for item in potencial_itens)
                    if peso_potencial > self.capacidade_mochila:
                        continue  # Pula se violar a capacidade

                    # Calcula o lucro potencial
                    _, _, _, _, lucro_potencial = self.funcao_objetivo.calcular_lucro_com_itens(
                        rota_atual, potencial_itens
                    )

                    if not is_tabu or (is_tabu and lucro_potencial > melhor_lucro):
                        vizinhos.append({
                            "rota": rota_atual,
                            "itens": potencial_itens,
                            "lucro": lucro_potencial,
                            "movimento": movimento_itens,
                            "tipo_movimento": "itens"
                        })

            # Tenta remover um item
            for item in solucao_atual_itens:
                movimento_itens = ("remove", item)
                is_tabu = self.tabu_lista_itens.get(movimento_itens, 0) > 0

                potencial_itens = solucao_atual_itens[:]
                potencial_itens.remove(item)

                # Calcula o lucro potencial
                _, _, _, _, lucro_potencial = self.funcao_objetivo.calcular_lucro_com_itens(
                    rota_atual, potencial_itens
                )

                if not is_tabu or (is_tabu and lucro_potencial > melhor_lucro):
                    vizinhos.append({
                        "rota": rota_atual,
                        "itens": potencial_itens,
                        "lucro": lucro_potencial,
                        "movimento": movimento_itens,
                        "tipo_movimento": "itens"
                    })

            # Gera vizinhos de rota
            vizinhos_rota = self.gerar_vizinhos_rota_swap(rota_atual)
            for nova_rota, movimento_rota in vizinhos_rota:
                is_tabu = self.tabu_lista_rotas.get(movimento_rota, 0) > 0

                # Recalcula o lucro com a nova rota e itens atuais
                _, _, _, _, lucro_potencial = self.funcao_objetivo.calcular_lucro_com_itens(
                    nova_rota, solucao_atual_itens
                )

                if not is_tabu or (is_tabu and lucro_potencial > melhor_lucro):
                    vizinhos.append({
                        "rota": nova_rota,
                        "itens": solucao_atual_itens,
                        "lucro": lucro_potencial,
                        "movimento": movimento_rota,
                        "tipo_movimento": "rota"
                    })

            # Se nenhum vizinho foi encontrado, encerra a busca
            if not vizinhos:
                break

            # Seleciona o melhor vizinho
            melhor_vizinho = max(vizinhos, key=lambda x: x["lucro"])

            # Atualiza as listas tabu e a solução atual
            if melhor_vizinho["tipo_movimento"] == "rota":
                movimento_rota = melhor_vizinho["movimento"]
                self.tabu_lista_rotas[movimento_rota] = self.duracao_tabu
                rota_atual = melhor_vizinho["rota"]
            elif melhor_vizinho["tipo_movimento"] == "itens":
                movimento_itens = melhor_vizinho["movimento"]
                self.tabu_lista_itens[movimento_itens] = self.duracao_tabu
                solucao_atual_itens = melhor_vizinho["itens"]

            # Diminui as durações tabu
            self.atualizar_tabu_lists()

            # Atualiza a melhor solução encontrada
            if melhor_vizinho["lucro"] > melhor_lucro:
                melhor_rota = rota_atual[:]
                melhor_itens = solucao_atual_itens[:]
                melhor_lucro = melhor_vizinho["lucro"]

        return melhor_rota, melhor_itens, melhor_lucro

def executar_busca_tabu(capacidade_mochila, duracao_tabu, func_obj, rota_inicial, itens_selecionados_iniciais, itens, num_iteracoes):
    """
    Função auxiliar para executar a Busca Tabu.
    Retorna um dicionário com os resultados.
    """
    try:
        busca_tabu = BuscaTabuRotaItens(
            capacidade_mochila,
            duracao_tabu,
            func_obj
        )
        
        # Executar a Busca Tabu
        melhor_rota, melhor_itens, melhor_lucro = busca_tabu.otimizar_mochila_e_rota_tabu(
            rota_inicial,
            itens_selecionados_iniciais,
            itens,
            num_iteracoes
        )
        
        # Recalcular valores finais com a melhor rota e itens
        valor_total, peso_total, tempo_viagem, aluguel_total, lucro_total = func_obj.calcular_lucro_com_itens(
            melhor_rota, melhor_itens
        )
        
        return {
            "melhor_rota": melhor_rota,
            "melhor_itens": melhor_itens,
            "melhor_lucro": melhor_lucro,
            "lucro_total": lucro_total,
            "tempo_viagem": tempo_viagem,
            "aluguel_total": aluguel_total,
            "valor_itens": valor_total,
            "peso_total": peso_total,
            "status": "Success"
        }
        
    except Exception as e:
        # Capturar exceções e retornar uma mensagem de erro
        return {
            "melhor_rota": [],
            "melhor_itens": [],
            "melhor_lucro": None,
            "lucro_total": None,
            "tempo_viagem": None,
            "aluguel_total": None,
            "valor_itens": None,
            "peso_total": None,
            "status": f"Error: {str(e)}"
        }

def main():
    # -----------------------------
    # Etapa 1: Pré-processamento
    # -----------------------------
    inicio_total = time.perf_counter()

    # Caminho para os arquivos de instância TTP
    arquivos_instancias = [
        "Instancias\\eil51_n50_bounded-strongly-corr_01.ttp"
    ]

    """ "Instancias\\eil51_n50_bounded-strongly-corr_01.ttp",
        "Instancias\\eil51_n150_bounded-strongly-corr_01.ttp",
        "Instancias\\eil51_n250_bounded-strongly-corr_01.ttp",
        "Instancias\\eil51_n50_uncorr-similar-weights_01.ttp",
        "Instancias\\eil51_n150_uncorr-similar-weights_01.ttp",
        "Instancias\\eil51_n250_uncorr-similar-weights_01.ttp",
        "Instancias\\eil51_n50_uncorr_01.ttp",
        "Instancias\\eil51_n150_uncorr_01.ttp",
        "Instancias\\eil51_n250_uncorr_01.ttp",
        "Instancias\\pr152_n151_bounded-strongly-corr_01.ttp",
        "Instancias\\pr152_n151_uncorr-similar-weights_01.ttp",
        "Instancias\\pr152_n151_uncorr_01.ttp"
        "Instancias\\a280_n279_bounded-strongly-corr_01.ttp", """
    
    """ "Instancias\\eil51_n50_bounded-strongly-corr_01.ttp",
        "Instancias\\eil51_n150_bounded-strongly-corr_01.ttp",
        "Instancias\\eil51_n250_bounded-strongly-corr_01.ttp",
        "Instancias\\eil51_n50_uncorr-similar-weights_01.ttp",
        "Instancias\\eil51_n150_uncorr-similar-weights_01.ttp",
        "Instancias\\eil51_n250_uncorr-similar-weights_01.ttp",
        "Instancias\\eil51_n50_uncorr_01.ttp",
        "Instancias\\eil51_n150_uncorr_01.ttp",
        "Instancias\\eil51_n250_uncorr_01.ttp",
        "Instancias\\pr152_n151_bounded-strongly-corr_01.ttp",
        "Instancias\\pr152_n453_bounded-strongly-corr_01.ttp",
        "Instancias\\pr152_n151_uncorr-similar-weights_01.ttp",
        "Instancias\\pr152_n453_uncorr-similar-weights_01.ttp",
        "Instancias\\pr152_n151_uncorr_01.ttp"
        "Instancias\\pr152_n453_uncorr_01.ttp",
        "Instancias\\a280_n279_bounded-strongly-corr_01.ttp",
        "Instancias\\a280_n279_uncorr-similar-weights_01.ttp",
        "Instancias\\a280_n279_uncorr_01.ttp" # Atualize com o caminho correto"""
    
    # Lista para armazenar todos os resultados de todas as instâncias
    todos_resultados = []

    iteracoes_list = [500, 500, 500, 500]
    duracao_tabu_list = [60, 3]
    
    for arquivo_instancia in arquivos_instancias:
        print(f"Iniciando instância: {arquivo_instancia}")
        inicio = time.perf_counter()

        # Inicialize a classe de pré-processamento
        preprocess = PreProcessingInstancia(arquivo=arquivo_instancia)

        # Acesse os dados necessários
        distancia_cidades = preprocess.distancia_cidades
        capacidade_mochila = preprocess.capacidade_mochila
        velocidade_minima = preprocess.velocidade_minima
        velocidade_maxima = preprocess.velocidade_maxima
        taxa_aluguel_por_tempo = preprocess.taxa_aluguel_por_tempo
        itens = preprocess.itens_associados  # Lista de tuplas (cidade, valor, peso)

        # -----------------------------
        # Etapa 2: Resolver o TSP
        # -----------------------------

        # Caminho para o executável do LKH-3
        caminho_executavel_lkh = "./LKH-3.exe"  # Atualize com o caminho correto do executável LKH-3

        # Inicialize o solver do TSP
        tsp_solver = TSPSolver(
            matriz_distancias=distancia_cidades,
            caminho_executavel=caminho_executavel_lkh,
            nome_problema="exemplo_tsp"
        )

        # Resolver o TSP
        tsp_solver.resolver(timeout=120)

        # Acessar a rota otimizada
        rota_inicial = tsp_solver.rota_otimizada

        # -----------------------------
        # Etapa 3: Iniciar Metaheurísticas a partir da Rota Inicial
        # -----------------------------

        # Instanciar FuncaoObjetivo e ItemSelector
        func_obj = FuncaoObjetivo(
            capacidade_mochila,
            taxa_aluguel_por_tempo,
            velocidade_maxima,
            velocidade_minima,
            distancia_cidades
        )

        item_selector = ItemSelector(
            capacidade_mochila
        )

        # Selecionar itens iniciais usando o ItemSelector
        itens_selecionados_iniciais = item_selector.selecionar_itens(
            rota_inicial,
            itens
        )

        # Calcular o lucro inicial e outras variáveis usando FuncaoObjetivo
        valor_total_inicial, peso_total_inicial, tempo_total_inicial, aluguel_total_inicial, lucro_inicial = func_obj.calcular_lucro_com_itens(
            rota_inicial, itens_selecionados_iniciais
        )

        fim_antesTabu = time.perf_counter()
        tempo_total_antesTabu = fim_antesTabu - inicio

        print(f"Tempo gasto Antes Tabu: {tempo_total_antesTabu:.2f} segundos")
        print(f"Lucro solução Inicial: {lucro_inicial}")

        # INICIANDO TESTES DE PARÂMETROS TABU
        for num_iteracoes in iteracoes_list:
            for duracao_tabu in duracao_tabu_list:
                print(f"Testando parâmetros - Iterações: {num_iteracoes}, Duração Tabu: {duracao_tabu}")
                
                inicio_tabu = time.perf_counter()

                # Preparar os argumentos para a função auxiliar
                args = (
                    capacidade_mochila,
                    duracao_tabu,
                    func_obj,
                    rota_inicial,
                    itens_selecionados_iniciais,
                    itens,
                    num_iteracoes
                )
                
                # Usar ProcessPoolExecutor para executar com timeout
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    future = executor.submit(executar_busca_tabu, *args)
                    try:
                        # Definir timeout de 600 segundos (30 minutos)
                        resultado_busca = future.result(timeout=1800)
                        status = resultado_busca.get("status", "Success")
                    except concurrent.futures.TimeoutError:
                        # Se exceder o tempo, cancelar o futuro e marcar como Timeout
                        future.cancel()
                        resultado_busca = {
                            "melhor_rota": [],
                            "melhor_itens": [],
                            "melhor_lucro": None,
                            "lucro_total": None,
                            "tempo_viagem": None,
                            "aluguel_total": None,
                            "valor_itens": None,
                            "peso_total": None,
                            "status": "Timeout"
                        }
                        status = "Timeout"
                        print(f"A busca tabu para Iterações: {num_iteracoes}, Duração Tabu: {duracao_tabu} excedeu o tempo limite e foi interrompida.")

                # Adicionar o nome da instância ao resultado
                resultado_busca["arquivo_instancia"] = arquivo_instancia
                resultado_busca["num_iteracoes"] = num_iteracoes
                resultado_busca["duracao_tabu"] = duracao_tabu

                # Calcular e adicionar 'tempo_total' somente se não for Timeout
                if status == "Success":
                    fim = time.perf_counter()
                    tempo_total = fim - inicio_tabu
                    resultado_busca["tempo_total"] = tempo_total
                else:
                    # Se for Timeout ou erro, definir 'tempo_total' como None ou outra marcação
                    resultado_busca["tempo_total"] = None

                # Se a busca não teve sucesso, preencher os campos com None ou valores padrão
                if status != "Success":
                    resultado_busca.update({
                        "melhor_rota": [],
                        "melhor_itens": [],
                        "melhor_lucro": None,
                        "lucro_total": None,
                        "tempo_viagem": None,
                        "aluguel_total": None,
                        "valor_itens": None,
                        "peso_total": None
                    })

                # Adicionar os resultados à lista geral
                todos_resultados.append(resultado_busca)

                # Imprimir o resultado no console
                if status == "Success":
                    print(f"Iterações: {num_iteracoes}, Duração Tabu: {duracao_tabu}, Lucro: {resultado_busca['melhor_lucro']}, Tempo: {resultado_busca['tempo_total']:.2f}s")
                elif status == "Timeout":
                    print(f"Iterações: {num_iteracoes}, Duração Tabu: {duracao_tabu}, Status: Timeout")
                else:
                    print(f"Iterações: {num_iteracoes}, Duração Tabu: {duracao_tabu}, Status: {status}")

        # -----------------------------
        # Etapa 4: Salvar Todos os Resultados em um Único Arquivo CSV
        # -----------------------------

        # Converter todos 
        # os resultados para um DataFrame
        df_todos = pd.DataFrame(todos_resultados)

        # Definir o nome do arquivo final
        nome_arquivo_final = r'resultados/Resultados_TTP.csv'

        # Salvar todos os resultados em um único arquivo CSV
        df_todos.to_csv(nome_arquivo_final, sep=";", index=False)

        fim_total = time.perf_counter()
        tempo_total_execucao = fim_total - inicio_total

        print(f"Tempo total de execução: {tempo_total_execucao:.2f} segundos")
        print(f"Todos os resultados foram salvos em {nome_arquivo_final}")

        # Opcional: Salvar também um arquivo .txt com logs adicionais
        nome_arquivo_log = "Registro_Experimentos.txt"
        with open(nome_arquivo_log, "w") as f:
            f.write(f"Tempo total de execução: {tempo_total_execucao:.2f} segundos\n")
            f.write(f"Resultados foram salvos em {nome_arquivo_final}\n\n")
            f.write("Detalhes dos resultados:\n")
            for res in todos_resultados:
                f.write(f"{res}\n")

        print(f"Registro dos experimentos salvo em {nome_arquivo_log}")

if __name__ == "__main__":
    main()
