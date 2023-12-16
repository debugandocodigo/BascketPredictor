import cv2  # pip install opencv-python
import cvzone  # pip install cvzone
from cvzone.ColorModule import ColorFinder
import numpy as np  # pip install numpy

video = cv2.VideoCapture('vid (3).mp4')  # Abrir o video

my_color_finder = ColorFinder(False)  # Instanciar o objeto ColorFinder
# Valores HSV para a cor vermelha
hsv_values = {'hmin': 8, 'smin': 96, 'vmin': 115, 'hmax': 14, 'smax': 255, 'vmax': 255}

points_list = []  # Lista de pontos
points_listX = []  # Lista de pontos X
points_listY = []  # Lista de pontos Y
x_list = [item for item in range(0,1300)]  # Lista de pontos X para desenhar a parabola

while True:
    _, frame = video.read()  # Ler o frame
    img = frame[0:900, :] # Cortar o frame
    img_color, mask = my_color_finder.update(img, hsv_values)  # Atualizar o objeto ColorFinder
    img_contour, contours = cvzone.findContours(img, mask, minArea=500)  # Encontrar os contornos
    if contours:
        # cx,cy = contours[0]['center']
        points_list.append(contours[0]['center'])
        points_listX.append(contours[0]['center'][0])
        points_listY.append(contours[0]['center'][1])

    if points_listX:  # Se a lista de pontos X não estiver vazia
        coeff = np.polyfit(points_listX, points_listY,2)  # Calcular os coeficientes da parabola
        for point in points_list:  # Percorrer a lista de pontos
            cv2.circle(img_contour, point,10,(0,255,0), cv2.FILLED)  # Desenhar um circulo em cada ponto

        for x in x_list:  # Percorrer a lista de pontos X
            poly = np.poly1d(coeff)  # Criar a parabola
            y = int(poly(x))  # Calcular o ponto Y
            # Desenhar um circulo em cada ponto da parabola
            cv2.circle(img_contour, (x,y), 2, (255, 0, 255), cv2.FILLED)
            if x >=330 and x <=410 and y >=580 and y<=610:  # Se o ponto estiver dentro do retangulo
                cv2.rectangle(img_contour, (550,600),(850,660), (0,255,0),-1)
                cv2.putText(img_contour, 'ACERTOU', (560,650), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255),5)

    cv2.imshow('Video', img_contour)
    cv2.waitKey(100)