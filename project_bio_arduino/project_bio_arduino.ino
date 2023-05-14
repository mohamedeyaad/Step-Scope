#include <ESP8266WiFi.h>
#include "Wire.h"
#include <MPU6050_light.h>

const int S0 = 13;    
const int S1 = 12;    
const int S2 = 14; 

const int fsr = A0;
int fsr_values[5];
int IMU_Angle1 = 0;
MPU6050 mpu(Wire);
unsigned long timer = 0;

float factor = 0.6;  //fsr_ignoring_factor
int threshold = 600; //fsr_threshold

int port = 5055;
WiFiServer server(port);

const char* ssid = "iPhone";
const char* password = "momo1234";

int ledPin = LED_BUILTIN; // GPIO13---D7 of NodeMCU

void setup() {
  Serial.begin(9600);
  
  /////////////////////// IMU INITIALIZATION /////////////////////////////
  Wire.begin();
  
  byte status = mpu.begin();
  Serial.print(F("MPU6050 status: "));
  Serial.println(status);
  while(status!=0){ } // stop everything if could not connect to MPU6050
  
  Serial.println(F("Calculating offsets, do not move MPU6050"));
  delay(1000);
  // mpu.upsideDownMounting = true; // uncomment this line if the MPU6050 is mounted upside-down
  mpu.calcOffsets(); // gyro and accelero
  Serial.println("Done!\n");
  /////////////////////////////////////////////////////////////////////////

  /////////////////////// MUX INITIALIZATION /////////////////////////////
  pinMode(S0, OUTPUT);
  pinMode(S1, OUTPUT);
  pinMode(S2, OUTPUT);
  digitalWrite(S0, LOW);
  digitalWrite(S1, LOW);
  digitalWrite(S2, LOW);
  /////////////////////////////////////////////////////////////////////////

  
  /*Serial.begin(9600);
  Serial.println();
  pinMode(ledPin, OUTPUT);
  digitalWrite(ledPin, LOW);
*/
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    delay(500);
  }
  
  Serial.println("");
  Serial.println("WiFi connected");

  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());

  // Start the server
  server.begin();
  Serial.println("Server started");
}

void loop() {
  // Check if a client has connected
  WiFiClient client = server.available();
  if (client) {
    if (client.connected())
    {
      Serial.println("CLient Connected");
    }
  while (client.connected()){
    /*while(client.available()>0){
      Serial.write(client.read());
    }*/
    //while(Serial.available()>0){
      //client.write(Serial.read());
      //client.print(Serial.readStringUntil('\n'));
    //}
  /////////////////////// IMU CALCULATIONS ///////////////////////////////
  mpu.update();

  if((millis()-timer)>10){ // print data every 10ms
  IMU_Angle1 =  mpu.getAngleX();
  //IMU_Angle1 =  mpu.getAngleY();
  //IMU_Angle1 =  mpu.getAngleY();

  Serial.print("Angle1: ");
  Serial.println(IMU_Angle1);

  timer = millis();  
  }
  /////////////////////////////////////////////////////////////////////////

  /////////////////////// FSR CALCULATIONS ///////////////////////////////

  /*READ FSR 1*/
  digitalWrite(S0, LOW);
  digitalWrite(S1, LOW);
  digitalWrite(S2, LOW);

  fsr_values[0] = 1025 - analogRead(fsr);

  if( fsr_values[0]<=threshold){
    fsr_values[0]=(fsr_values[0]*factor)+1;
  }

  /*READ FSR 2*/
  digitalWrite(S0, HIGH);
  digitalWrite(S1, LOW);
  digitalWrite(S2, LOW);

  fsr_values[1] = 1025 - analogRead(fsr);

  if( fsr_values[1]<=threshold){
    fsr_values[1]=(fsr_values[1]*factor)+1;
  }
  
  /*READ FSR 3*/
  digitalWrite(S0, LOW);
  digitalWrite(S1, HIGH);
  digitalWrite(S2, LOW);

  fsr_values[2] = 1025 - analogRead(fsr);

  if( fsr_values[2]<=threshold){
    fsr_values[2]=(fsr_values[2]*factor)+1;
  }
  
  /*READ FSR 4*/
  digitalWrite(S0, HIGH);
  digitalWrite(S1, HIGH);
  digitalWrite(S2, LOW);

  fsr_values[3] = 1025 - analogRead(fsr);

  if( fsr_values[3]<=threshold){
    fsr_values[3]=(fsr_values[3]*factor)+1;
  }
  
  /*READ FSR 5*/
  digitalWrite(S0, LOW);
  digitalWrite(S1, LOW);
  digitalWrite(S2, HIGH);

  fsr_values[4] = 1025 - analogRead(fsr);

  if( fsr_values[4]<=threshold){
    fsr_values[4]=(fsr_values[4]*factor)+1;
  }
  
  Serial.print("FSR readings :[");

  Serial.print(fsr_values[0]);
  Serial.print(",");
    
  Serial.print(fsr_values[1]);
  Serial.print(",");

  Serial.print(fsr_values[2]);
  Serial.print(",");

  Serial.print(fsr_values[3]);
  Serial.print(",");
  
  Serial.print(fsr_values[4]);
  Serial.println("]");
  client.print((String)" Connection Successful, FSR readings = ["+fsr_values[2]+","+fsr_values[0]+","+fsr_values[1]+","+fsr_values[4]+","+fsr_values[3]+"] imu 1 value = "+IMU_Angle1+" imu 2 value = "+2);
  /////////////////////////////////////////////////////////////////////////
  delay(100);
  }
  client.stop();
  Serial.println("Client disconnected");
  }
}
