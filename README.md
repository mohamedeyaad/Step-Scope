# Step-Scope

<div align="center">
  <img src="https://github.com/mohamedeyaad/Step-Scope/blob/GUI/gui/graphics/LOGO_color.png" alt="Step Scope Logo" width="300"/>
  <br>
  
  [![Watch the Demo](https://img.youtube.com/vi/HEZB3_pBwzg/0.jpg)](https://www.youtube.com/watch?v=HEZB3_pBwzg)
  
  *Click the image above to watch the project demo video.*
</div>

## 📋 Project Overview
Step Scope is a project developed as part of the Biomedical Engineering Course at the Faculty of Engineering, Ain Shams University. The main objective of this project is to design and develop a wireless, portable, and wearable system that utilizes inertial measurement units (IMUs) and force-sensitive resistors (FSRs) to monitor foot plantar pressure and gait in individuals with diabetic foot disease (DFD) or other neurological abnormalities affecting gait.

Please note that the system has been thoroughly tested on both healthy subjects and patients/simulated patients to ensure its effectiveness and accuracy.

## ✨ Key Features
The system provides the following features:
- **Real-time visualization** of foot pressure using color coding (Heatmap).
- **Measurement** of various spatiotemporal gait parameters.
- **Detection** of gait events and phases (Heel Strike, Mid-Stance, Toe Off, etc.).
- **Analysis** of shank kinematics using IMUs.
- **Estimation** of the center of pressure (CoP).
- **Assessment** of patient stability based on FSR and IMU data.
- **Wireless Telemetry:** Data is transmitted wirelessly to a PC for visualization, analysis, and reporting.

---

## ⚙️ Hardware Architecture

![Hardware Setup](https://github.com/mohamedeyaad/Step-Scope/blob/GUI/Hardware.png)

The system is built around the **ESP8266 (NodeMCU)** microcontroller, which collects data from the sensor array and transmits it via TCP/IP to the ROS backend.

### Components
* **Microcontroller:** NodeMCU v1.0 (ESP8266)
* **IMU:** 2x MPU6050 (Accelerometer & Gyroscope for Thigh and Shin)
* **Force Sensors:** 5x Force Sensitive Resistors (FSR) placed at key plantar pressure points.
* **Multiplexer:** 8-channel Analog Multiplexer (to handle 5 FSRs on the single ESP8266 Analog Pin).

### Pinout Configuration
| Component | NodeMCU Pin | GPIO | Description |
| :--- | :--- | :--- | :--- |
| **Multiplexer S0** | D7 | 13 | Channel Select Bit 0 |
| **Multiplexer S1** | D6 | 12 | Channel Select Bit 1 |
| **Multiplexer S2** | D5 | 14 | Channel Select Bit 2 |
| **Multiplexer Sig**| A0 | ADC | Analog Signal Input |
| **MPU6050 SDA** | D2 | 4 | I2C Data |
| **MPU6050 SCL** | D1 | 5 | I2C Clock |

---

## 💻 Software Architecture

The PC-side software utilizes **ROS (Robot Operating System)** to modularize data handling, processing, and visualization.

1.  **Communication Node (`comm_data.py`):** Establishes a TCP/IP socket connection with the ESP8266.
    * Publishes raw sensor strings to the `/comm` topic.
2.  **Parser Nodes:**
    * `fsr_data.py`: Subscribes to `/comm`, parses FSR arrays, and publishes to `/fsr_data`.
    * `imu_pub.py`: Parses IMU angles and publishes to `/imu_data`.
3.  **Visualization Node (`project.py`):** A **Tkinter**-based GUI that subscribes to all sensor topics.
    * Visualizes Heatmaps, Knee Angles, and Gait Phases in real-time using OpenCV and Matplotlib.

---

## 🖥️ Graphical User Interface (GUI)

The GUI allows for patient data entry, real-time monitoring, and data recording.

![GUI Screenshot](https://github.com/mohamedeyaad/Step-Scope/blob/GUI/GUI.jpeg)

### Gait Phase Detection Example
Below is an example of the system detecting the "Mid Stance" phase during a live test.

![Mid Stance Example](https://github.com/mohamedeyaad/Step-Scope/blob/GUI/Example%20-%20Mid%20Stance.jpeg)

---

## 🚀 Getting Started

### Prerequisites
* **OS:** Ubuntu 20.04 (Recommended for ROS Noetic)
* **ROS Version:** ROS Noetic Desktop Full
* **Python:** Python 3.8+

### Installation
1.  **Clone the repository into your Catkin Workspace:**
    ```bash
    cd ~/catkin_ws/src
    git clone https://github.com/mohamedeyaad/Step-Scope.git
    cd ..
    catkin_make
    source devel/setup.bash
    ```

2.  **Install Python Dependencies:**
    ```bash
    pip3 install -r requirements.txt
    ```
    *(Note: Ensure `numpy`, `matplotlib`, `scipy`, `pandas`, `XlsxWriter`, `opencv-python`, and `Pillow` are installed).*

3.  **Upload Firmware:**
    * **Important:** The firmware code is located in a separate branch.
    * Switch to the firmware branch:
      ```bash
      git checkout Arduino-Code
      ```
    * Open the `.ino` file in Arduino IDE.
    * Configure your WiFi credentials in the code.
    * Upload to the NodeMCU.

### Running the System
1.  **Power on the Hardware:** Ensure the NodeMCU is powered and connected to the WiFi.
2.  **Launch the System:**
    Run the ROS launch file to start the communication, parsing, and GUI nodes simultaneously.
    ```bash
    roslaunch step_scope nodes.launch
    ```

## 📧 Contact
For any inquiries, please contact me at: **mooeyad@gmail.com**
