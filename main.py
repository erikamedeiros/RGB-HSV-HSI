import cv2
import numpy as np
import sys
import os
import argparse
import convert
import math
from matplotlib import pyplot as plt

#função responsável por extrair as informações do sashibo
def test(location):

    #abrindo imagem
    src = cv2.imread(location)
    #redimensionando a imagem para 600x400
    src = cv2.resize(src, (600, 400))

    #alterando espaço de cores para HSV e jogando a nova imagem na variável hsv
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

    #detectando a cor do sashibo por range de cor HSV (vermelho mais escuro até o vermelho mais claro)
    #Obs: Foram realizados vários testes com ranges diferentes de vermelho para se chegar nesses valores abaixo

    #definindo os valores HSV mínimo para detecção de cor
    lower_range = np.array([150, 40, 40])
    #definindo os valores HSV máximos para detecção de cor
    upper_range = np.array([189, 255, 255])

    #criando uma máscara na ára onde não for encontrada a cor que estiver no range definido acima
    mask = cv2.inRange(hsv, lower_range, upper_range)

    #imprimindo a máscara
    #cv2.imshow('mask', mask)

    #foi passado um filtro gausiano para suavizar as bordas da área do sashibo para minimizar as perdas
    gray = cv2.GaussianBlur(mask, (7, 7), 3)

    #aqui convertemos a imagem para bits, onde de o valor do pixel for menos que o limite
    # ele se torna 0 e se for maior se torna 1. Isso ajuda no algoritmo de contornos.
    t, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)

    #contornando a área do sashibo
    contours, a = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #pintando contornos na imagem para visualização
    #cv2.drawContours(src, contours, -1, (0, 0, 255), 1, cv2.LINE_AA)

    #variáveis responsáveis por armazenar os valores da posição do sashibo na imagem original
    x = 0
    y = 0
    w = 0
    h = 0

    #Percorrendo todo contorno para extrair as informações da posição do sashibo
    for c in contours:
        area = cv2.contourArea(c)
        if area > 10 and area < 1000000:
            #criando um retângulo na área encontrada e passando as informações de ponto e área para as variáveis
            (x, y, w, h) = cv2.boundingRect(c)
            #cv2.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 2, cv2.LINE_AA)
            #print(x,y,w,h)
            break

    #extraindo sashibo
    crop = src[y:y+h, x:x+w]

    #abaixo só trabalhamos com a imagem que foi extraída (que é o sahsibo):

    #padrão usado pelo opencv, vamos usar nas plotagens dos gráficos abaixo
    color = ('b','g','r')

    #plotando gráficos:

    plt.figure(figsize=(6, 4))

    plt.suptitle('Extração de informações do ' + location)

    grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)

    plt.subplot(grid[0, 0])
    plt.title('Sample')
    plt.imshow(crop)

    plt.subplot(grid[0, 1:])
    plt.title('Histogram')

    #calculando histograma para o RGB
    for i,col in enumerate(color):
        histr = cv2.calcHist([crop], [i], None, [256], [0, 256])
        plt.plot(histr, color = col)
        plt.xlim([0, 256])

    plt.subplot(grid[1, :2])
    plt.title('HSV')

    x = np.arange(3)

    #capturando dados HSV da imagem separados
    hue = crop[:, :, 0].mean()
    saturation = crop[:, :, 1].mean()
    value = crop[:, :, 2].mean()

    values = [saturation, hue, value]

    plot = plt.bar(x, values)

    #plotando HSV em gráfico
    for value in plot:
        height = value.get_height()
        plt.text(value.get_x() + value.get_width()/2., 1.002*height,'%d' % int(height), ha='center', va='bottom')

    plt.xticks(x, ('Saturation', 'Hue', 'Value'))
    plt.xlabel("HSV parameters")
    plt.ylabel(" ") 


    #convertendo RGB para HSI
    hsi = convert.RGB_TO_HSI(crop)

    #cv2.imshow('HSI Image', hsi)

    #capturando dados HSI da imagem separados
    # Geralmente o componente h esta entre 0 e 2pi
    # os componentes s e i estão entre 0 e 1
    hue = hsi[:, :, 0].mean()
    saturation = hsi[:, :, 1].mean() 
    intensity = hsi[:, :, 2].mean() 

    if math.isnan(float(hue)):
        hue = 0

    values1 = [float(saturation), float(hue), float(intensity)]

    print(values1)

    plt.subplot(grid[1, 2])
    plt.title('HSI')

    plt.xticks(x, ('Saturation', 'Hue', 'Intensity'))
    plt.xlabel("HSI parameters")
    plt.ylabel("  ") 

    x = np.arange(3)
    plot = plt.bar(x, values1)

    for value in plot:
        height = value.get_height()
        plt.text(value.get_x() + value.get_width()/2., 1.002*height,'%f' % float(height), ha='center', va='bottom')

    #cv2.imshow('contornos', src)

    #cv2.imshow('crop', crop)

    #print(hue)
    #print(saturation)

    plt.gcf().canvas.set_window_title(location)

    plt.show(block=False)

    return 0

def listDir(arg):
    files = []
    for a, _, arq in os.walk(arg):
        if len(arq) != 0:
            for file in arq:
                files.append(a + "/" + file)
    return files
    

#função responsável por coletar os argumentos (nome dos aquivos de imagem) passados na chamada do programa 
def main(argv):

    print('Processando imagens...')

    print(argv[1])

    files = listDir(argv[1])

    print(files)

    for n in files:
        test(n)

    print('<<< Processado >>>')

    #pausando aplicação para manter as janelas dos gráficos abertas
    while True:
        key = cv2.waitKey(1)
        #se clicar ESC o programa fecha (caso n feche é só encerrar o processo pelo terminal (Control + C))
        if key == 27:
            cv2.destroyAllWindows()
    return 0


#chamada da função main
if __name__ == '__main__':
    sys.exit(main(sys.argv))
