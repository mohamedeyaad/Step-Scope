#!/usr/bin/env python3

# Importing needed libraries
from tkinter import *
from PIL import ImageTk,Image
import re
import cv2
import time
import rospy
from std_msgs.msg import String
from std_msgs.msg import Int32 , Float32MultiArray
import scipy.signal as signal
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import pandas as pd

# Defining the paths
path = "/home/mooeyad//catkin_ws/src/biomedical_project/gui/graphics"
path_saving = "project/saved_data"

# Defining and setting parameters for the root GUI window
root = Tk()
root.title("StepScope")   #root window title
root.geometry("700x360")  #Startup Resolution
#root.iconbitmap(path+"/icon_c.ico") #Program icon


# Load all images that will be used
HeatMap_cv = cv2.imread(path+"/heatmap.png")
HeatMapBlue_cv = cv2.imread(path+"/heatmap_flatblue.png")
HeatMapMASK_cv = cv2.imread(path+"/mask.png", cv2.IMREAD_GRAYSCALE)


# Convert the image to a PIL Image object
HeatMap_pil = Image.fromarray(cv2.cvtColor(HeatMap_cv, cv2.COLOR_BGRA2RGBA))


# Create a Tkinter PhotoImage object from the PIL Image object
HeatMap_tk = ImageTk.PhotoImage(HeatMap_pil, format="RGBA")

# Importing the Gait Cycle Images
gaitNA_tk = ImageTk.PhotoImage(Image.open(path+"/GaitPhases/NA.png"), format="RGBA")
gaitIC_tk = ImageTk.PhotoImage(Image.open(path+"/GaitPhases/IC.png"), format="RGBA")
gaitFF_tk = ImageTk.PhotoImage(Image.open(path+"/GaitPhases/FF.png"), format="RGBA")
gaitMS_tk = ImageTk.PhotoImage(Image.open(path+"/GaitPhases/MS.png"), format="RGBA")
gaitHL_tk = ImageTk.PhotoImage(Image.open(path+"/GaitPhases/HL.png"), format="RGBA")
gaitTO_tk = ImageTk.PhotoImage(Image.open(path+"/GaitPhases/TO.png"), format="RGBA")


# importing and visualizing the logo on the root window
logo_img = ImageTk.PhotoImage(Image.open(path+"/LOGO_colorR.png")) 
logo_label = Label(image = logo_img)
logo_label.grid(row=0,column=0)

###################  VARIABLES INITIALIZATION #################
P_Name = ""                                   # Patient Name
P_Gender = ""                                 # Patient Gender
P_Age = 0                                     # Patient Age
P_Height = 0                                  # Patient Height
P_Weight = 0                                  # Patient Weight

offsetX= 1                                    # X-axis offset for grid allignment
offsetY= 0                                    # Y-axis offset for grid allignment

DataConfirmed = False                         # Boolean for data confirmation
Transitioned = False                          # Boolean for window transitioning

record_data = False

StopWatchRunning = False                      # Boolean for to run the stop watch

IMU_Shin = 0                                  # IMU Shin Readings
IMU_Thigh = 0                                 # IMU Thigh Reading
Pressure_Readings = [0,0,0,0,0]               # FSR Readings

GenderChoices = ["Male","Female"]             # Gender Choices for drop down menu

GaitStates = ["N/A","Initial Contact","Foot Flat","Midstance","Heel lift" , "Toe off", "Swing" ]
                                              # Gait Phases List
    
gaitphases_tk = [gaitNA_tk , gaitIC_tk , gaitFF_tk , gaitMS_tk , gaitHL_tk , gaitTO_tk]
                                              # Gait Phases Images List 

NewWindow = None                              # Variable to store new window

stopwatch_label = None                        # Stopwatch label

index = 0                                     # Index to access the previous lists

FSR5_loc = (28,57)                            # FSR5 pixel location
FSR4_loc = (25,112)                           # FSR4 pixel location
FSR3_loc = (88, 158)                          # FSR3 pixel location
FSR2_loc = (66, 233)                          # FSR2 pixel location
FSR1_loc = (42 ,290)                          # FSR1 pixel location

COP = [50,170]					# CENTER OF PRESSURE
FSR = [1,1,1,1,1]				# FSR Reading list
######################################################################

# Funtion for a number only string
def contains_only_numbers(input_string):
    pattern = "^[0-9]+$"  # match string with only numbers
    return not(bool(re.match(pattern, input_string)))

# Function to clear a grid area
def clear_grid_area(row, column , window):
    for widget in window.grid_slaves(row, column):
        widget.destroy()

def cv2_imshow(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.show()

#Stop watch functionlity 
def start_stopwatch():
    global start_time
    global StopWatchRunning
    
    start_time = time.time() # get the current time when the stopwatch is started
    StopWatchRunning = True
    update_stopwatch() # start updating the stopwatch label

def update_stopwatch():
    global NewWindow
    global stopwatch_label
    global StopWatchRunning
    global update_id
    
    if StopWatchRunning:
        elapsed_time = time.time() - start_time # calculate the elapsed time
        #clear_grid_area(7,0,NewWindow)
        stopwatch_label = Label(NewWindow, text=format_time(elapsed_time), font=("Arial", 12))
        stopwatch_label.grid(row=7,column=0)
        update_id =stopwatch_label.after(100, update_stopwatch) # schedule the update function to be called after 100ms (0.1 second)
    else:
        pass

def format_time(elapsed_time):
    minutes = int(elapsed_time / 60)
    seconds = int(elapsed_time - minutes * 60)
    milliseconds = int((elapsed_time - minutes * 60 - seconds) * 100)
    return f"{minutes:02d}:{seconds:02d}:{milliseconds:02d}"

def stop_stopwatch():
    global StopWatchRunning

    stopwatch_label.after_cancel(update_id) # cancel the scheduled update function
    StopWatchRunning = False


#Simple Test Function for debugging

def test():
    Fsr_Data = [ 700 , 300 , 500 , 1000 , 800 ]
    UpdateHeatMap(Fsr_Data)
    UpdateIMU1Value(12)
    UpdateIMU1Value(14)
    
# Function that will excute from the confirm button
def ConfirmClick():
    global P_Name , P_Age , P_Height , P_Weight , P_Gender
    global DataConfirmed
    global root

    # Get the entered data in the text boxes
    P_Name= (NameEntry.get())
    P_Gender = GenderVar.get()
    P_Age = (AgeEntry.get())
    P_Height = (HeightEntry.get())
    P_Weight = (WeightEntry.get())
    
    # Check for the data validity 
    if (P_Name == "") or (P_Age == "") or (P_Height == "") or (P_Weight == "")\
    or contains_only_numbers(P_Age) or contains_only_numbers(P_Height) or \
    contains_only_numbers(P_Weight) :
        # Data entered is incomplete / invalid
        clear_grid_area(6+offsetX,0+offsetY,root)
        clear_grid_area(7+offsetX,0+offsetY,root)
        ErrorLabel = Label(root,text="Please Enter the data correctly")
        ErrorLabel.grid(row=6+offsetX,column=0+offsetY)
        DataConfirmed = False

    else:
        # Data entered is valid
        clear_grid_area(8+offsetX,0+offsetY,root)
        ConfirmationLabel = Label(root,text="Data is Entered")
        ConfirmationLabel.grid(row=8+offsetX,column=0+offsetY)
        P_StrData = "Patient name is "+P_Name+" age is " +str(P_Age)+ ", is " +\
        str(P_Height)+" cm tall and weighs "+str(P_Weight)+" KG " +"Gender is "+ P_Gender
        PatientDataLabel = Label(root,text=P_StrData)
        PatientDataLabel.grid(row=9+offsetX,column=0+offsetY)
        DataConfirmed = True 

from scipy.signal import butter, filtfilt


def butter_bandpass(lowcut, highcut, fs, order=5):
 nyq = 0.5 * fs
 low = lowcut / nyq
 high = highcut / nyq
 b, a = butter(order, [low, high], btype='band')
 return b, a
 
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):     	
 b, a = butter_bandpass(lowcut, highcut, fs, order=order)
 y = filtfilt(b, a, data)
 return y

# Function for the filter choices
def apply_filter(data, selected_filter):
    # Get the selected filtering option from the dropdown menu
    #selected_filter = var.get()
    
    if selected_filter == "raw data":
            return data
                  
    # Apply the selected filtering option to the recorded signals
    elif selected_filter == "Low Pass Filter":
        # Apply a low pass filter
            cutoff_freq = 10 # Set the cutoff frequency of the filter
            sampling_freq = 100 # Set the sampling frequency of the data
            # Calculate the filter coefficients using a Butterworth filter
            b, a = signal.butter(5, cutoff_freq / (sampling_freq / 2), 'lowpass')
            # Apply the filter to the signal
            filtered_signal = signal.filtfilt(b, a, data)
     
            return filtered_signal

    elif selected_filter == "Box Filter":
        # Apply a box filter (moving average)
            window_size = 5 # Set the window size of the filter
            window = np.ones(window_size)/float(window_size)
            filtered_signal = np.convolve(data, window, 'same')
            return filtered_signal
            

    elif selected_filter == "Band Pass Filter":
     # Define bandpass filter parameters
     lowcut = 10  # Hz
     highcut = 20  # Hz 
     fs = 1000  # Hz
     order = 4
     # Apply bandpass filter to data
     b, a = butter(order, [lowcut, highcut], btype='band', fs=fs)
     filtered_data = filtfilt(b, a, data)
     return filtered_data
"""
        # Apply a band pass filter
        # Set the cut-off frequencies and filter order
        fs = 100  # Sample rate, Hz
        f_high = 30  # High cut-off frequency, Hz
        order = 5  # Filter order
        
        # Calculate filter coefficients using a Butterworth filter
        b, a = signal.butter(order, [f_low / (fs / 2), f_high / (fs / 2)], 'bandpass')
        
        # Apply the filter to the data
        filtered_data = signal.filtfilt(b, a, data)
"""
        
# Function that will excute from the next button        
def NextClick():

    global DataConfirmed
    global Transitioned
    global NewWindow
    global root

    # Check for Confirmed data
    if DataConfirmed:
        # Data is Confirmed
        clear_grid_area(6+offsetX,0+offsetY,root)
        clear_grid_area(7+offsetX,0+offsetY,root)
        TransitionLabel = Label(root,text="Transitioning to the next stage....")
        TransitionLabel.grid(row=6+offsetX,column=0+offsetY)
        # Start a new window for data display
        DataWindow = Toplevel(root)
        DataWindow.title = ("Data Display") 
        DataWindow.geometry("800x600")
        NewWindow = DataWindow
        DrawDataWindow(DataWindow)
        Transitioned = True
        
    else:
        # Data is not Confirmed yet
        clear_grid_area(6+offsetX,0+offsetY, root)
        clear_grid_area(7+offsetX,0+offsetY, root)
        ErrorLabel = Label(root,text="Please Enter the data and confirm it first")
        ErrorLabel.grid(row=6+offsetX,column=0+offsetY)
        
# Creating entry text boxes 
NameEntry = Entry(root)
AgeEntry = Entry(root)
HeightEntry = Entry(root)
WeightEntry = Entry(root)

# Creating a drop menu for the gender choice
GenderVar=StringVar(root)
GenderVar.set("Male")
GenderChoice = OptionMenu(root , GenderVar ,*GenderChoices)

# Global labels to configure later
StatusLabel1 = 0 
StatusLabel2 = 0
IMU1ValueLabel = 0
IMU2ValueLabel = 0
KneeAngleValueLabel = 0
GaitPhaseValueLabel = 0
Gaitimg_Label = 0
HeatMapImg_Label = 0

# Creating a Label for the required text
NameLabel   =  Label(root,text="Patient Name:")
AgeLabel    =  Label(root,text="Patient Age:")
HeightLabel =  Label(root,text="Patient Hieght:")
WeightLabel =  Label(root,text="Patient Weight:")
GenderLabel =  Label(root,text="Patient Gender:")

# changes for the filters dropout window choices

filterOptionlabel = Label(root, text="Select a filtering option:")
filterOptionlabel.grid(row=5+offsetX,column=0+offsetY)  #must switch pack to grid two different methods of organization
options = ["raw data","Low Pass Filter", "Box Filter", "Band Pass Filter"]
var = StringVar(root)
var.set(options[0])
dropdown = OptionMenu(root, var, *options)
dropdown.grid()  

# Create a button to apply the selected filter
result = StringVar() # create a StringVar to store the result
#button = Button(root, text="Apply Filter", command=lambda: result.set(apply_filter(data)))
button = Button(root, text="Apply Filter", command=lambda: None)
button.grid()   
label = Label(root, textvariable=result) # create a Label widget to display the result
label.grid()

# Creating the Confrim data button
ConfirmButton = Button(root, text="Confirm Data" , padx = 50 , command = ConfirmClick)

# Creating the next button
NextButton    = Button(root, text="Next" , padx = 50 , command = NextClick)

# Creating Update button (Debugging)
updatebutton  = Button(root, text="update" , padx = 50 , command = test)


# Assigning every label and button a place on the window grid
####################Labels###################
NameLabel.grid(row=0+offsetX,column=0+offsetY)
AgeLabel.grid(row=1+offsetX,column=0+offsetY)
HeightLabel.grid(row=2+offsetX,column=0+offsetY)
WeightLabel.grid(row=3+offsetX,column=0+offsetY)
GenderLabel.grid(row=4+offsetX,column=0+offsetY)

####################Entries###################
NameEntry.grid(row=0+offsetX,column=1+offsetY)
AgeEntry.grid(row=1+offsetX,column=1+offsetY)
HeightEntry.grid(row=2+offsetX,column=1+offsetY)
WeightEntry.grid(row=3+offsetX,column=1+offsetY)
GenderChoice.grid(row=4+offsetX,column=1+offsetY)

####################Buttons###################
ConfirmButton.grid(row=5+offsetX,column=0+offsetY)
NextButton.grid(row=5+offsetX,column=1+offsetY)
updatebutton.grid(row=7,column=1)


# Function to record the sent values and patient data in an np array and save it to the disk as csv file when stop record is clicked
# It must start a timer when clicked to show the recording time
def RecordClick():
    
    global NewWindow
    global stopwatch_label
    global record_data
    record_data = True
    
    stopwatch_label = Label(NewWindow, text="00:00:00", font=("Arial", 12))
    stopwatch_label.grid(row=7,column=0)
    start_stopwatch()
    clear_grid_area(6,1,NewWindow)
    # Creating Start Record button 
    stopbutton  = Button(NewWindow, text="stop record" , padx = 50 , command = StopClick)
    stopbutton.grid(row=6,column=1)


def StopClick():
    
    global NewWindow
    global record_data
    record_data = False
    
    # Define the arrays and patient data
    global sensor_imu1, sensor_imu2, sensor_fsr_1, sensor_fsr_2, sensor_fsr_3, sensor_fsr_4, sensor_fsr_5, COP_x, COP_y
    
    sensor_imux1 = np.array(sensor_imu1)
    sensor_imux2 = np.array(sensor_imu2)
    sensor_fsr_x1 = np.array(sensor_fsr_1)
    sensor_fsr_x2 = np.array(sensor_fsr_2)
    sensor_fsr_x3 = np.array(sensor_fsr_3)
    sensor_fsr_x4 = np.array(sensor_fsr_4)
    sensor_fsr_x5 = np.array(sensor_fsr_5)
    
    
    rospy.loginfo("Target========================")
    rospy.loginfo(sensor_fsr_x5)
    # Define the array names and file name
    array_names = ["sensor_imu1", "sensor_imu2", "sensor_fsr_1", "sensor_fsr_2", "sensor_fsr_3", "sensor_fsr_4", "sensor_fsr_5"]
    file_name = "patientreading.xlsx"
    # Write the arrays to an Excel file named "patientreading.xlsx"
    file_name = write_arrays_to_excel([sensor_imux1, sensor_imux2, sensor_fsr_x1,sensor_fsr_x2,sensor_fsr_x3,sensor_fsr_x4,sensor_fsr_x5], array_names,
    file_name=file_name)
    
    stop_stopwatch()
    clear_grid_area(6,1,NewWindow)
    # Creating Stop Record button
    SavingLabel = Label(NewWindow,text="File saved to path: "+ path_saving)
    SavingLabel.grid(row=6,column=1)
    
    #Apply the filter using the selected option
    selected_filter = var.get()
    rospy.loginfo(selected_filter)
    data_1 = sensor_fsr_x1
    filtered_data_1 = apply_filter(data_1, selected_filter)
    rospy.loginfo(filtered_data_1)
    
    data_2 = sensor_fsr_x2
    filtered_data_2 = apply_filter(data_2, selected_filter)
    rospy.loginfo(filtered_data_2)
    
    data_3 = sensor_fsr_x3
    filtered_data_3 = apply_filter(data_3, selected_filter)
    rospy.loginfo(filtered_data_3)
    
    data_4 = sensor_fsr_x4
    filtered_data_4 = apply_filter(data_4, selected_filter)
    rospy.loginfo(filtered_data_4)
    
    data_5 = sensor_fsr_x5
    filtered_data_5 = apply_filter(data_5, selected_filter)
    rospy.loginfo(filtered_data_5)
    
    data_y = COP_y
    filtered_data_y = apply_filter(data_y, selected_filter)
    rospy.loginfo(filtered_data_y)
    
    #Update the result label
    #result.set(str(filtered_data))
    
    # Plot the original and filtered signals
    
    # Create a 2x3 grid of subplots
    fig, axs = plt.subplots(3, 2)
       
    #t = np.linspace(0, 1, 1000)
    t = np.arange(0, len(data_1))
    

    
    # Plot data in each subplot
    axs[0, 0].plot(t, data_1, label = 'Original Signal')
    axs[0, 0].plot(t, filtered_data_1, label='Filtered Signal')
    axs[0, 0].set_title('FSR_data1')
    axs[0, 1].plot(t, data_2, label = 'Original Signal')
    axs[0, 1].plot(t, filtered_data_2, label='Filtered Signal')
    axs[0, 1].set_title('FSR_data2')
    axs[1, 0].plot(t, data_3, label = 'Original Signal')
    axs[1, 0].plot(t, filtered_data_3, label='Filtered Signal')
    axs[1, 0].set_title('FSR_data3')
    axs[1, 1].plot(t, data_4, label = 'Original Signal')
    axs[1, 1].plot(t, filtered_data_4, label='Filtered Signal')
    axs[1, 1].set_title('FSR_data4')
    axs[2, 0].plot(t, data_5, label = 'Original Signal')
    axs[2, 0].plot(t, filtered_data_5, label='Filtered Signal')
    axs[2, 0].set_title('FSR_data5')
    axs[2, 1].plot(t, data_y, label = 'Original Signal')
    axs[2, 1].plot(t, filtered_data_y, label='Filtered Signal')
    axs[2, 1].set_title('COP_y')
    
    # Remove the last subplot since it's not being used
    #fig.delaxes(axs[2, 1])
    # Add overall title to the figure
    fig.suptitle('Plots')
    # Adjust spacing between subplots
    fig.tight_layout()
    # Display the figure
    plt.show()

    #fig, ax = plt.subplots()
    #ax.plot(t, data, label='Original signal')
    #ax.plot(t, filtered_data, label='Filtered signal')
    #ax.legend()
    #ax.set_xlabel('Time (s)')
    #ax.set_ylabel('Amplitude')
    
    
    """
    #ax.set_title('Original and filtered signals')
    plt.plot(t, data, label='Original Signal')
    plt.plot(t, filtered_data, label='Filtered Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()
    """
# Function that draws the content of the new display data windows for the first time
def DrawDataWindow(DataWindow):

    global StatusLabel1
    global StatusLabel2
    global IMU1ValueLabel
    global IMU2ValueLabel
    global KneeAngleValueLabel
    global GaitPhaseValueLabel
    global Gaitimg_Label
    global HeatMapImg_Label
    
    # Connection Status Labels and Grid assignment
    ConnectionLabel1 = Label(DataWindow ,text="Connection 1:")
    ConnectionLabel2 = Label(DataWindow ,text="Connection 2:")
    ConnectionLabel1.grid(row=0,column=0)
    ConnectionLabel2.grid(row=1,column=0)
    StatusLabel1 = Label(DataWindow ,text="Failed" ,fg ='#ff0000')
    StatusLabel2 = Label(DataWindow ,text="Failed" ,fg='#ff0000')
    StatusLabel1.grid(row=0,column=1)
    StatusLabel2.grid(row=1,column=1)
    
    # IMU sensors readings Labels and Grid assignment
    IMU1Label = Label(DataWindow ,text="IMU Thigh:")
    IMU2Label = Label(DataWindow ,text="IMU Shin:")
    IMU1Label.grid(row=2,column=0)
    IMU2Label.grid(row=3,column=0)
    IMU1ValueLabel = Label(DataWindow ,text=str(IMU_Thigh))
    IMU2ValueLabel = Label(DataWindow ,text=str(IMU_Shin))
    IMU1ValueLabel.grid(row=2,column=1)
    IMU2ValueLabel.grid(row=3,column=1)
    
    # Knee Angle calculation and Grid assingment
    KneeAngleLabel = Label(DataWindow ,text="Knee angle:")
    KneeAngleLabel.grid(row=4,column=0)
    KneeAngleValueLabel = Label(DataWindow ,text=str(abs(IMU_Thigh-IMU_Shin)))
    KneeAngleValueLabel.grid(row=4,column=1)
    
    # Gait Phase prediction Labels and Grid assignment
    GaitPhaseLabel = Label(DataWindow ,text="Predicted Gait Phase:")
    GaitPhaseLabel.grid(row=5,column=0)
    GaitPhaseValueLabel = Label(DataWindow ,text=GaitStates[index],fg='#ff0000')
    GaitPhaseValueLabel.grid(row=5,column=1)
    Gaitimg_Label = Label(DataWindow , image=gaitphases_tk[index])
    Gaitimg_Label.grid(row=9,column=0)
    
    # Heatmap Label and Grid assignment
    HeatMapImg_Label = Label(DataWindow , image=HeatMap_tk)
    HeatMapImg_Label.grid(row=9,column=1)
    
    # Record button and Grid assignment
    Recordbutton = Button(DataWindow, text="Record" , padx = 50 , command= RecordClick)
    Recordbutton.grid(row=6,column=0)

def map_color (reading):
    #Readings come from range 1 - 1023
    if reading < 512 :
        map_color = [255-(reading/2),0+(reading/2),0]
    else:
        reading_inv = abs((reading-1023)/2)
        map_color = [0,0+(reading_inv),255-(reading_inv)]
    return map_color

def map_reading(img):
    # Get image dimensions
    height, width = img.shape[:2]

    # Create a mask for the pixels that meet the color criteria
    mask1 = (img[..., 0] < 255) & (img[..., 1] > 0) & (img[..., 2] == 0)
    mask2 = (img[..., 0] == 0) & (img[..., 1] < 255) & (img[..., 2] > 0)
    mask = mask1 | mask2

    # Apply the color mapping to the pixels that meet the criteria
    readings = np.zeros((height, width), dtype=np.uint32)
    if mask.any():
        b = img[..., 0][mask].astype(np.int32)
        g = img[..., 1][mask].astype(np.int32)
        r = img[..., 2][mask].astype(np.int32)

        readings[mask] = np.select(
            [mask1[mask], mask2[mask]],
            [np.where(g < b, g - b + 255, 2 * g - 2 * b),
             np.where(r < g, 2 * g + 2 * r + 255, 4 * r - 4 * g)],
            default=0
        )

    return readings


def box_filter(arr, kernel_size):
    kernel = np.ones((kernel_size, 1), dtype=np.float32) / kernel_size
    arr_filtered = np.apply_along_axis(lambda m: np.convolve(m, kernel.flatten(), mode='same'), axis=1, arr=arr)
    arr_filtered = np.apply_along_axis(lambda m: np.convolve(m, kernel.flatten(), mode='same'), axis=0, arr=arr_filtered)
    return arr_filtered
    
def Calculate_COP(arr):
    sumX = FSR1_loc[0]*arr[0] + FSR2_loc[0]*arr[1] + FSR3_loc[0]*arr[2] + FSR4_loc[0]*arr[3] + FSR5_loc[0]*arr[4]
    sumY = FSR1_loc[1]*arr[0] + FSR2_loc[1]*arr[1] + FSR3_loc[1]*arr[2] + FSR4_loc[1]*arr[3] + FSR5_loc[1]*arr[4]
    W = arr[0]+arr[1] + arr[2] + arr[3] + arr[4]
    copX = sumX/W
    copY = sumY/W
    return (int(copX),int(copY))

#Update functions to update each value in the new display data window
def UpdateStatus1(state):        #Update Connection 1
    global NewWindow
    global StatusLabel1

    if state:
        StatusLabel1.config ( text="Successful" ,fg ='#00ff00')
        #StatusLabel1.grid(row=0,column=1)
    else:
        StatusLabel1.config (text="Failed" ,fg ='#ff0000')
        #StatusLabel1.grid(row=0,column=1)
    
def UpdateStatus2(state):        #Update Connection 2
    global NewWindow
    global StatusLabel2

    if state:
        StatusLabel2.config ( text="Successful" ,fg ='#00ff00')
        #StatusLabel1.grid(row=1,column=1)
    else:
        StatusLabel2.config (text="Failed" ,fg ='#ff0000')
        #StatusLabel1.grid(row=1,column=1)

def UpdateIMU1Value(Value):        #Update IMU Thigh readings
    global NewWindow
    global IMU1ValueLabel
    
    IMU1ValueLabel.config(text=str(Value))
    IMU_Thigh = Value
    #IMU1ValueLabel.grid(row=2,column=1)
    
def UpdateIMU2Value(Value):        #Update IMU Shin readings
    global NewWindow
    global IMU2ValueLabel
    
    IMU2ValueLabel.config(text=str(Value))
    IMU_Shin = Value
    #IMU2ValueLabel.grid(row=3,column=1)

def UpdateKneeAnglee():            #Update Knee Angle calculation
    global NewWindow
    global KneeAngleValueLabel
    
    KneeAngleValueLabel.config(text=str(abs(IMU_Thigh-IMU_Shin)))
    #KneeAngleValueLabel.grid(row=4,column=1)

def UpdateGaitPhase():        #Update the predicted gait phase
    global NewWindow
    global COP
    global FSR
    global GaitPhaseValueLabel
    global Gaitimg_Label
    
    y_val = COP[1]
    if FSR[0] < 400 and FSR[1] < 400 and FSR[2] < 400 and FSR[3] < 400 and FSR[4] < 400 :
    	index = 6
    elif  y_val < 120: 
    	#TOE OFF 
    	index = 5
    elif y_val < 160 and y_val > 120:
    	#HEEL LIFT
    	index = 4
    elif y_val < 190 and y_val > 160:
    	#MID STANCE
    	index = 3
    elif y_val < 210 and y_val > 190:
       #FOOT FLAT
    	index = 2 	
    elif y_val > 210:
    	#HEEL CONTACT
    	index = 1
    	
    else:
    	#find from knee angle
    	pass
    
    GaitPhaseLabel = Label(NewWindow ,text="Predicted Gait Phase:")
    GaitPhaseLabel.grid(row=5,column=0)
    GaitPhaseValueLabel.config(text=GaitStates[index],fg='#ff00ff')
    if index < 6:
    	Gaitimg_Label.config(image=gaitphases_tk[index])
    else:
    	Gaitimg_Label.config(image=gaitphases_tk[0])

def UpdateHeatMap(FSR_Arr):         #Update the heat map
    global NewWindow
    global root 
    global HeatMap_cv
    global HeatMapMASK_cv
    global HeatMapImg_Label
    global COP
    
    HeatMap_cvCPY = HeatMapBlue_cv.copy()
    HeatMap_cvCPY2 = HeatMapBlue_cv.copy()
    
    FSR1_color = map_color(FSR_Arr[0])
    FSR2_color = map_color(FSR_Arr[1])
    FSR3_color = map_color(FSR_Arr[2])
    FSR4_color = map_color(FSR_Arr[3])
    FSR5_color = map_color(FSR_Arr[4])
    
    HeatMap_cvCPY = cv2.circle(HeatMap_cvCPY, FSR1_loc , 35 , FSR1_color , -1)
    HeatMap_cvCPY = cv2.circle(HeatMap_cvCPY, FSR2_loc , 35 , FSR2_color , -1)
    HeatMap_cvCPY = cv2.circle(HeatMap_cvCPY, FSR3_loc , 35 , FSR3_color , -1)
    HeatMap_cvCPY = cv2.circle(HeatMap_cvCPY, FSR4_loc , 35 , FSR4_color , -1)
    HeatMap_cvCPY = cv2.circle(HeatMap_cvCPY, FSR5_loc , 35 , FSR5_color , -1)
    
    # Get image dimensions
    height, width = HeatMapMASK_cv.shape[:2]
    
    # create an empty NumPy array of size 0 with dtype int
    wpv =  np.zeros((height, width , 3), dtype=np.int64)
    values =  np.zeros((height, width , 3), dtype=np.int64)
    
    # Iterate over each pixel and print its value
    # Set threshold and RGB lower bound
    threshold = 200
    rgb_lower_bound = [50, 50, 50]

    # Create boolean masks for pixels meeting the criteria
    mask1 = HeatMapMASK_cv > threshold
    mask2 = (HeatMap_cvCPY > rgb_lower_bound).all(axis=2)

    # Create new array for output
    wpv = np.zeros_like(HeatMap_cvCPY)

    # Assign new values based on masks
    wpv[mask1 & mask2] = [255, 0, 0]
    wpv[mask1 & ~mask2] = HeatMap_cvCPY[mask1 & ~mask2]
    wpv[~mask1] = [0, 0, 0]
    
    values = map_reading(wpv)
    
    values = box_filter(values,42)
    
    for i in range(height):
        for j in range(width):
            if values[i,j] > 0 and HeatMapMASK_cv[i, j] > 200 :
                HeatMap_cvCPY2[i,j] = map_color(values[i,j])
    
    # Convert the image to a PIL Image object
    COP = Calculate_COP(FSR_Arr)
    HeatMap_cvCPY2 = cv2.circle(HeatMap_cvCPY2, COP , 9 , [0,0,0] , -1)
    HeatMap_pil = Image.fromarray(cv2.cvtColor(HeatMap_cvCPY2, cv2.COLOR_BGR2RGB))
    window = Tk()  # This line must come BEFORE creating ImageTk
    # Create a Tkinter PhotoImage object from the PIL Image object
    HeatMap_tk = ImageTk.PhotoImage(HeatMap_pil)
    
    # Heatmap Label and Grid assignment
    HeatMapImg_Label.config(image=HeatMap_tk)
    #HeatMapImg_Label.grid(row=9,column=1)
    HeatMapImg_Label.img = HeatMap_tk
    window.destroy()



#looping the gui
def callback_status(msg):
	global Transitioned
	global NewWindow
	
	#rospy.loginfo(Transitioned)	

	if msg.data == "Successful" and Transitioned == True:
	 UpdateStatus1(True)
	 UpdateStatus2(True)
	elif Transitioned:
	 UpdateStatus1(False)
	 UpdateStatus2(False)
	 
# Initialize an empty list to store sensor readings
sensor_imu1 = []

def callback_imu(msg):
	global Transitioned
	global NewWindow
	global sensor_imu1
	global IMU_Shin        
	global record_data      
	   
	#rospy.loginfo("ana hna")
	if Transitioned == True:
		IMU_Shin = msg.data
		UpdateIMU1Value(msg.data)
		#UpdateGaitPhase()
		UpdateKneeAnglee()
		if record_data == True:
			sensor_imu1.append(msg.data) # Store the sensor reading in the list
		

# Initialize an empty list to store sensor readings
sensor_imu2 = []


def callback_imu2(msg):
	global Transitioned
	global NewWindow
	global IMU_Thigh 
	global sensor_imu2
	global record_data
	
	if Transitioned == True:
	      IMU_Thigh = msg.data
	      UpdateIMU2Value(msg.data)
	      #UpdateGaitPhase()	
	      UpdateKneeAnglee()
	      if record_data == True: 
	      	sensor_imu2.append(msg.data) # Store the sensor reading in the list
	
# Initialize empty lists to store sensor readings
sensor_fsr_1 = []
sensor_fsr_2 = []
sensor_fsr_3 = []
sensor_fsr_4 = []
sensor_fsr_5 = [] 
COP_x=[]
COP_y=[]

def callback_fsr(data):
	global Transitioned
	global sensor_fsr_1
	global sensor_fsr_2
	global sensor_fsr_3
	global sensor_fsr_4
	global sensor_fsr_5   
	global record_data
	global FSR
	global COP_x
	global COP_y 
	global COP


	if Transitioned == True:
		#array = data.data
		#UpdateHeatMap(array)
		#rospy.loginfo(array)
		#data = np.random.rand(20)
		#data = np.random.randint(1, 1025, size=50)
		UpdateHeatMap(data.data)
		FSR = data.data
		UpdateGaitPhase()
		if record_data == True:
			#sensor_array = np.array(data.data)
			#rospy.loginfo(sensor_array)
			#sensor_data.append(sensor_array) # Store the sensor array in the list
			# Store each element of the sensor array in a separate list
			sensor_fsr_1.append(data.data[0])
			sensor_fsr_2.append(data.data[1])
			sensor_fsr_3.append(data.data[2])
			sensor_fsr_4.append(data.data[3])
			sensor_fsr_5.append(data.data[4])
			COP_x.append(-(COP[0]-50))
			COP_y.append(-(COP[1]-170))
			rospy.loginfo(sensor_fsr_3)

def write_arrays_to_excel(arrays, array_names, file_name):
    global P_Name , P_Age , P_Height , P_Weight , P_Gender
    
    writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
    # Create a worksheet for patient data
    patient_sheet = writer.book.add_worksheet('Patient Data')
    patient_data = np.array([P_Name, P_Age, P_Height, P_Weight, P_Gender])
    # Write the patient data to the worksheet
    patient_sheet.write('A1', 'Name')
    patient_sheet.write('B1', 'Age')
    patient_sheet.write('C1', 'Height')
    patient_sheet.write('D1', 'Weight')
    patient_sheet.write('E1', 'Gender')
    patient_sheet.write('A2', patient_data[0])
    patient_sheet.write('B2', patient_data[1])
    patient_sheet.write('C2', patient_data[2])
    patient_sheet.write('D2', patient_data[3])
    patient_sheet.write('E2', patient_data[4])
    for i in range(len(arrays)):
        pd.DataFrame(arrays[i]).to_excel(writer, sheet_name=array_names[i], index=False, header=False)
        worksheet = writer.sheets[array_names[i]]
        # Write the patient data to the worksheet
        worksheet.write('A1', 'Name')
        worksheet.write('B1', 'Age')
        worksheet.write('C1', 'Height')
        worksheet.write('D1', 'Weight')
        worksheet.write('E1', 'Gender')
        worksheet.write('A2', patient_data[0])
        worksheet.write('B2', patient_data[1])
        worksheet.write('C2', patient_data[2])
        worksheet.write('D2', patient_data[3])
        worksheet.write('E2', patient_data[4])
    writer.save()
    return file_name




if __name__ == '__main__':
	try:
		rospy.init_node('callback_node', anonymous=True)
		rospy.Subscriber("status", String, callback_status)
		rospy.Subscriber("imu_data", Int32, callback_imu)
		rospy.Subscriber("imu_data2", Int32, callback_imu2)
		rospy.Subscriber("fsr_data", Float32MultiArray, callback_fsr)
		while not rospy.is_shutdown():
			root.mainloop()
			rospy.sleep(1)
	except rospy.ROSInterruptException:
		pass

