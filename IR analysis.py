import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

#### INPUT ####
# folder that contains datapoints
folderName = '2dIR'

#### SETTINGS ####
# settings listed below are suitable for 2D data

# intensity of noise filtering; higher values mean more blurring
medianKernel = 5

# blurring radius in x and y direction; higher values mean more blurring; note: these values need to be odd
gaussian_x = 9
gaussian_y = 181

# decay blurring strength; higher values mean blurring will be more focused on the center
gaussianSigma = 60

# number of pixels that are averaged on both sides when iterating over each pixel in a row
pixelsAveraged1 = 10

# number of pixels that are averaged on both sides when iterating over pixels closer to the leading edge; this number should be smaller than pixelsAveraged1 since higher precision is needed
pixelsAveraged2 = 6

# vertical range of pixels considered when determining transition line; range is selected so that noise at the root and the tip is disregarded
rangeVer = (40, 400)

# maximal fraction of standard deviation for the point to be included during filtering
maxStd = 0.9

# minimal fraction of the points left after filtering for the line to be considered as transition line
minFiltered = 0.5

# critical angle at which the line closest to the leading edge is considered to be the transition line
criticalAngle = 7.5

# margin of averaged pixels between the leading edge and detected transition points
margin = 2

# minimal average difference of the more aft lines to be considered as transition line
minDifference1 = 4.68

# minimal average difference of the more forward lines to be considered as transition line
minDifference2 = 3.1

# width of the cropped image
width = 360

# settings listed below are suitable for 3D data

# medianKernel = 5
# gaussian_x = 9
# gaussian_y = 181
# gaussianSigma = 60
# pixelsAveraged1 = 9
# pixelsAveraged2 = 6
# rangeVer = (40, 400)
# maxStd = 1.5
# minFiltered = 0.5
# criticalAngle = 9.5
# margin = 2
# minDifference1 = 3.84
# minDifference2 = 3.1
# width = 360


# processing image
def findTransition(data, angle):
    # removing NaN values from the array
    data = data[:, ~np.isnan(data).all(axis=0)]
    # normalising data
    data = ((data - np.amin(data)) / (np.amax(data) - np.amin(data)) * 255)
    # converting to pixel data
    data = data.astype(np.uint8)

    # processing data using median and gaussian blur
    blurred = cv2.medianBlur(data, medianKernel)
    blurred = cv2.GaussianBlur(blurred, (gaussian_x, gaussian_y), gaussianSigma)

    # creating empty arrays to store locations of edges and potential transitions
    edges = np.zeros((len(blurred), 2), dtype=int)
    edge = (0, 0)
    differencesVer = np.zeros((len(blurred), 3))
    transitions1 = np.zeros((len(blurred), 2), dtype=int)
    transitions2 = np.zeros((len(blurred), 2), dtype=int)

    # iterating over each row of pixels
    for i in range(len(blurred)):

        # iterating over each pixel in a row and calculating differences between pixels to the right and to the left
        differencesHor1 = np.zeros(len(blurred[i]))
        for j in range(len(blurred[i])):
            if j - pixelsAveraged1 >= 0 and j + pixelsAveraged1 <= len(blurred[i]):
                differencesHor1[j] = np.absolute(np.average(blurred[i, j - pixelsAveraged1:j]) - np.average(blurred[i, j:j + pixelsAveraged1]))

        # selecting two locations where differences are the highest
        edges[i, 0] = np.argmax(differencesHor1)
        for j in range(len(differencesHor1)):
            if differencesHor1[j] > differencesHor1[edges[i, 1]] and np.absolute(edges[i, 0] - j) > pixelsAveraged1:
                edges[i, 1] = j
        edges = np.sort(edges, axis=1)

        # averaging the detected locations to determine position of the edges
        edge = int(np.average(edges[rangeVer[0]:rangeVer[1], 0])), int(np.average([edges[rangeVer[0]:rangeVer[1], 1]]))

        # iterating over each pixel between edges and calculating differences between pixels to the right and to the left
        differencesHor1 = np.zeros(len(blurred[i]))
        for j in range(len(blurred[i])):
            if edges[i, 0] + 2 * pixelsAveraged1 <= j <= edges[i, 1] - margin * pixelsAveraged1:
                differencesHor1[j] = np.absolute(np.average(blurred[i, j - pixelsAveraged1:j]) - np.average(blurred[i, j:j + pixelsAveraged1]))

        # selecting two locations where differences are the highest
        transitions1[i, 0] = np.argmax(differencesHor1)
        for j in range(len(differencesHor1)):
            if differencesHor1[j] > differencesHor1[transitions1[i, 1]] and np.absolute(transitions1[i, 0] - j) > 3 * pixelsAveraged1:
                transitions1[i, 1] = j
        transitions1 = np.sort(transitions1, axis=1)

        # iterating over pixels closer to the leading edge and calculating differences between pixels to the right and to the left
        differencesHor2 = np.zeros(len(blurred[i]))
        for j in range(len(blurred[i])):
            if edges[i, 0] + 10 * pixelsAveraged2 <= j <= edges[i, 1] - pixelsAveraged2:
                differencesHor2[j] = np.absolute(np.average(blurred[i, j - pixelsAveraged2:j]) - np.average(blurred[i, j:j + pixelsAveraged2]))

        # selecting two locations where differences are the highest
        transitions2[i, 0] = np.argmax(differencesHor2)
        for j in range(len(differencesHor2)):
            if differencesHor2[j] > differencesHor2[transitions2[i, 1]] and np.absolute(transitions2[i, 0] - j) > pixelsAveraged2:
                transitions2[i, 1] = j
        transitions2 = np.sort(transitions2, axis=1)

        # saving maximal horizontal differences to calculate vertical differences
        differencesVer[i, 0] = differencesHor1[transitions1[i, 0]]
        differencesVer[i, 1] = differencesHor1[transitions1[i, 1]]
        differencesVer[i, 2] = differencesHor2[transitions2[i, 0]]

    # cropping locations of transitions and vertical differences
    transitions1 = transitions1[rangeVer[0]:rangeVer[1], :]
    transitions2 = transitions2[rangeVer[0]:rangeVer[1], :]
    differencesVer = differencesVer[rangeVer[0]:rangeVer[1], :]

    # calculating average and standard deviation of the first detected transition line
    transitions1Avg = np.average(transitions1[:, 0])
    transitions1Std = np.std(transitions1[:, 0])

    # filtering locations that are too far from the average
    transitions1Filtered = []
    for i in range(len(transitions1)):
        if round(transitions1Avg - maxStd * transitions1Std) <= transitions1[i, 0] <= round(transitions1Avg + maxStd * transitions1Std):
            transitions1Filtered.append(transitions1[i, 0])

    # calculating average and standard deviation of the second detected transition line
    transitions2Avg = np.average(transitions1[:, 1])
    transitions2Std = np.std(transitions1[:, 1])

    # filtering locations that are too far from the average
    transitions2Filtered = []
    for i in range(len(transitions1)):
        if round(transitions2Avg - maxStd * transitions2Std) <= transitions1[i, 1] <= round(transitions2Avg + maxStd * transitions2Std):
            transitions2Filtered.append(transitions1[i, 1])

    # calculating average and standard deviation of the third detected transition line
    transitions3Avg = [np.average(transitions2[:, 0]), np.average(transitions2[:, 1])]
    transitions3Std = [np.std(transitions2[:, 0]), np.std(transitions2[:, 1])]

    # filtering locations that are too far from the average
    transitions3Filtered = []
    for i in range(len(transitions2)):
        if round(transitions3Avg[0] - maxStd * transitions3Std[0]) <= transitions2[i, 0] <= round(transitions3Avg[0] + maxStd * transitions3Std[0]) \
                and round(transitions3Avg[1] - maxStd * transitions3Std[1]) <= transitions2[i, 1] <= round(transitions3Avg[1] + maxStd * transitions3Std[1]):
            transitions3Filtered.append(np.average(transitions2[i, :]))

    # calculating the average of vertical differences for each transition line
    differences = np.zeros(3)
    differences[0] = np.average(differencesVer[:, 0])
    differences[1] = np.average(differencesVer[:, 1])
    differences[2] = np.average(differencesVer[:, 2])

    # choosing one of the three detected lines
    if differences[0] >= minDifference1 and len(transitions1Filtered) > minFiltered * (rangeVer[1] - rangeVer[0]) and angle < criticalAngle:
        transition = round(np.average(transitions1Filtered))
    elif differences[1] >= minDifference1 and len(transitions2Filtered) > minFiltered * (rangeVer[1] - rangeVer[0]) and angle < criticalAngle:
        transition = round(np.average(transitions2Filtered))
    elif differences[2] >= minDifference2:
        transition = round(np.average(transitions3Filtered))
    else:
        transition = edge[1]

    # printing parameters for debugging
    # print('Differences 1: ' + differences[0])
    # print('Differences 2: ' + differences[1])
    # print('Differences 3: ' + differences[2])
    # print('Length of filtered transitions 1:' + str(len(transitions1Filtered)))
    # print('Length of filtered transitions 1:' + str(len(transitions2Filtered)))
    # print('Length of filtered transitions 1:' + str(len(transitions3Filtered)))

    # calculating the location of transition as percentage of chord length
    XC = 1 - ((transition - edge[0]) / (edge[1] - edge[0]))

    # printing edges and transition line on the generated image
    for i in range(len(data)):
        data[i, edge[0] - 1:edge[0] + 1] = 0
        data[i, edge[1] - 1:edge[1] + 1] = 0
        data[i, transition - 1:transition + 1] = 0
        # data[i, edges[i, 0] - 1:edges[i, 0] + 1] = 0
        # data[i, edges[i, 1] - 1:edges[i, 1] + 1] = 0

    # printing detected lines on the generated image
    # for i in range(len(transitions1)):
    #     data[i + rangeVer[0], transitions1[i, 0] - 1:transitions1[i, 0] + 1] = 0
    #     data[i + rangeVer[0], transitions1[i, 1] - 1:transitions1[i, 1] + 1] = 0
    #     data[i + rangeVer[0], transitions2[i, 0] - 1:transitions2[i, 0] + 1] = 0
    #     data[i + rangeVer[0], transitions2[i, 1] - 1:transitions2[i, 1] + 1] = 0

    # calculating midpoint between edges and cropping the image
    midpoint = int((edge[1] - edge[0]) / 2 + edge[0])
    data = data[:, int(midpoint - width / 2):int(midpoint + width / 2)]
    blurred = blurred[:, int(midpoint - width / 2):int(midpoint + width / 2)]

    # converting data to contiguous array
    data = np.ascontiguousarray(data, dtype=np.uint8)

    # settings for placing AoA and transition location on the image
    text1 = 'AoA: ' + str(angle)
    text2 = 'x/c = ' + str(round(XC, 3))
    org1 = (60, 20)
    org2 = (60, 40)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    color = (255, 0, 0)
    thickness = 1

    # inserting text to the image
    data = cv2.putText(data, text1, org1, font, fontScale, color, thickness, cv2.LINE_AA)
    data = cv2.putText(data, text2, org2, font, fontScale, color, thickness, cv2.LINE_AA)

    # showing generated images
    # cv2.imshow("data", data)
    # cv2.imshow("blurred", blurred)
    # cv2.waitKey(0)

    # saving generated images
    path = 'Images'
    fileName = 'AoA=' + str(angle) + ',XC=' + str(round(XC, 3)) + '.jpg'
    cv2.imwrite(os.path.join(path, fileName), data)
    # cv2.imwrite(os.path.join(path, 'blurred.jpg'), blurred)

    return XC


# detecting all folders in the selected directory
folders = os.listdir(folderName + '/.')

# creating empty array for results
results = np.zeros((len(folders), 2))

# iterating over each folder
for i, folder in enumerate(folders):

    # detecting all files in the selected folder
    folderPath = folderName + '/' + folder + '/.'
    files = os.listdir(folderPath)

    # creating empty array in the size of data
    dataPoints = np.zeros((480, 640))

    # monitoring progress of the program
    print('---------------------------------------')
    print('Progress: ' + str(round(i / len(folders) * 100, 2)) + '%')
    print('AoA: ' + folder)

    # iterating over detected files
    for file in files:

        # importing data into array
        filePath = folderName + '/' + folder + '/' + file
        dataPoint = np.genfromtxt(filePath, delimiter=';')

        # removing NaN values from the array
        dataPoint = dataPoint[:, ~np.isnan(dataPoint).all(axis=0)]

        # adding imported data to the array
        dataPoints += dataPoint

        break

    # calculating average of the data
    # dataPoints = dataPoints / len(files)

    # calculating location of transition and saving it into the results
    transitionXC = findTransition(dataPoints, float(folder))
    results[i] = [float(folder), transitionXC]

# saving results to text file
results = results[results[:, 0].argsort()]
np.savetxt('results.txt', results, delimiter=',')

# generating graph of location vs angle of attack
plt.plot(results[:, 0], results[:, 1])
plt.xlabel("Angle of attack [deg]")
plt.ylabel("Location of transition [x/c]")
plt.show()
