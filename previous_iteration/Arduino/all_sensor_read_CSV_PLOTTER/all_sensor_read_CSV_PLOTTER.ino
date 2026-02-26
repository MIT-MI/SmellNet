/*
    Multichannel_gas_sensor_V2.0.ino
    Description: A terminal for Seeed Grove Multichannel gas sensor V2.0.
    2019 Copyright (c) Seeed Technology Inc.  All right reserved.
    Author: Hongtai Liu(lht856@foxmail.com)
    2019-9-29

    The MIT License (MIT)
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.1  USA
*/

#include <Multichannel_Gas_GMXXX.h>
#include <MQUnifiedsensor.h>
#include "Adafruit_BME680.h"

#include <math.h>

//Definitions
#define Board "Adafruit ESP32 Feather"
#define Voltage_Resolution 5
// #define type "MQ-135" //MQ135
#define ADC_Bit_Resolution 12 // For arduino UNO/MEGA/NANO
#define RatioMQ135CleanAir 3.6//RS / R0 = 3.6 ppm  
//#define calibration_button 13 //Pin to calibrate your sensor

#define         RatioMQ9CleanAir        (9.6) //RS / R0 = 60 ppm 
#define         RatioMQ3CleanAir        (60) //RS / R0 = 60 ppm 
#define         RatioMQ135CleanAir 3.6//RS / R0 = 3.6 ppm  

// R0s calculated during calibration in other file 
#define MQ135R0 14.29
#define MQ9R0 2.96
#define MQ3R0 0.04

// HCHO
#define Vc 4.95
//the number of R0 you detected just now
#define HCHOR0 10.1


// BME680
#define BME_SCK 18
#define BME_MISO 19
#define BME_MOSI 23
#define BME_CS 5
#define SEALEVELPRESSURE_HPA (1017)

Adafruit_BME680 bme(&Wire); // I2C

// cvs data collection constants
#define SAMPLING_FREQ_HZ 2
#define SAMPLE_COLLECTION_TIME_MS 6000*1 // 60000 = one minute

bool currCollecting = false;

// if you use the software I2C to drive the sensor, you can uncommnet the define SOFTWAREWIRE which in Multichannel_Gas_GMXXX.h.
#ifdef SOFTWAREWIRE
    #include <SoftwareWire.h>
    SoftwareWire myWire(3, 2);
    GAS_GMXXX<SoftwareWire> gas;
#else
    #include <Wire.h>
    GAS_GMXXX<TwoWire> gas;
#endif

static uint8_t recv_cmd[8] = {};


// digital pin where button is wired 
const int buttonPin = 33;
int buttonState = 0;  

// MQ-135

#define mq135pin A1 //Analog input 0 of your arduino
MQUnifiedsensor MQ135(Board, Voltage_Resolution, ADC_Bit_Resolution, mq135pin, "MQ-135");

// MQ-9
#define mq9pin A0
MQUnifiedsensor MQ9(Board, Voltage_Resolution, ADC_Bit_Resolution, mq9pin, "MQ-9");
#define         PreaheatControlPin5      (3) // Preaheat pin to control with 5 volts
#define         PreaheatControlPin14      (4) // Preaheat pin to control with 1.4 volts

#define mq3pin A3
MQUnifiedsensor MQ3(Board, Voltage_Resolution, ADC_Bit_Resolution, mq3pin, "MQ-3");


#define hchopin A2

unsigned long start_timestamp;

void setup() {

    Serial.println("test");
    Serial.begin(115200);
    // If you have changed the I2C address of gas sensor, you must to be specify the address of I2C.
    //The default addrss is 0x08;

    Serial.println("Setting up sensors...");

    pinMode(buttonPin, INPUT);

    // Multichannel
    gas.begin(Wire, 0x08); // use the hardware I2C
    //gas.begin(MyWire, 0x08); // use the software I2C
    //gas.setAddress(0x64); change thee I2C address

    // MQ-135
    MQ135.setRegressionMethod(1); //_PPM =  a*ratio^b
    MQ135.setA(77.255); MQ135.setB(-3.18 ); // Configure the equation to to calculate Alcohol concentration
    MQ135.init(); 
    MQ135.setR0(MQ135R0);

    // MQ-9

    pinMode(PreaheatControlPin5, OUTPUT);
    pinMode(PreaheatControlPin14, OUTPUT);
    //Set math model to calculate the PPM concentration and the value of constants
    MQ9.setRegressionMethod(1); //_PPM =  a*ratio^b
    MQ9.setA(599.65); MQ9.setB(-2.244); // Configure the equation to to calculate LPG concentration
    MQ9.init(); 
    MQ9.setR0(MQ9R0); 

    // MQ-3 
    MQ3.setRegressionMethod(1); //_PPM =  a*ratio^b
    MQ3.setA(4.8387); MQ3.setB(-2.68); // Configure the equation to to calculate Benzene concentration
    MQ3.init(); 
    MQ3.setR0(MQ3R0);

    //BME680
    if (!bme.begin()) {
    Serial.println("Could not find a valid BME680 sensor, check wiring!");
    while (1);
    }
     // Set up oversampling and filter initialization
    bme.setTemperatureOversampling(BME680_OS_8X);
    bme.setHumidityOversampling(BME680_OS_2X);
    bme.setPressureOversampling(BME680_OS_4X);
    bme.setIIRFilterSize(BME680_FILTER_SIZE_3);
    bme.setGasHeater(320, 150); // 320*C for 150 ms


    currCollecting = false;

    delay(10000*3);
    
}

void loop() {
    uint8_t len = 0;
    uint8_t addr = 0;
    uint8_t i;
    uint32_t val = 0;

    unsigned long timestamp;


    // printing header
    buttonState = digitalRead(buttonPin);

    // on press start collecting data
    if(!currCollecting)
    {
      // Serial.println("timestamp,NO2,C2H50H,VOC,CO,Alcohol,LPG,Benzene,HCHO,Temperature,Pressure,Humidity,Gas_Resistance,Altitude");
        Serial.println("timestamp,NO2,C2H50H,VOC,CO,Alcohol,LPG,Benzene,Temperature,Pressure,Humidity,Gas_Resistance,Altitude");
      currCollecting = true;
      delay(2000);
      start_timestamp = millis();
    }

    timestamp = millis()-start_timestamp;
     // check if sample collection time has finished
    if(timestamp > SAMPLE_COLLECTION_TIME_MS && currCollecting) // stop collecting csv data
    {
      Serial.println();
      currCollecting = false;
    }
    if(currCollecting)
    {
    // Serial.print(timestamp); Serial.print(",");
    // getting multichannel readings: 
    // detecting NO2, CSH50H, VOC, CO 
    val = gas.measure_NO2(); Serial.print("NO2:"); Serial.print(val); Serial.print(",");
    val = gas.measure_C2H5OH(); Serial.print("C2H50H:"); Serial.print(val); Serial.print(",");
    val = gas.measure_VOC(); Serial.print("VOC:"); Serial.print(val); Serial.print(",");
    val = gas.measure_CO(); Serial.print("CO:"); Serial.print(val); Serial.print(",");

    // MQ-135 detecting alcohol 
    Serial.print("Alcohol:");
    MQ135.update(); // Update data, the arduino will read the voltage from the analog pin
    Serial.print(MQ135.readSensor()); // Sensor will read PPM concentration using the model, a and b values set previously or from the setup
    // MQ135.serialDebug(); // Will print the table on the serial port
    Serial.print(",");

    // MQ-9 detecting LPG
    Serial.print("LPG:");
    MQ9.update(); // Update data, the arduino will read the voltage from the analog pin
    Serial.print(MQ9.readSensor()); // Sensor will read PPM concentration using the model, a and b values set previously or from the setup
    // MQ9.serialDebug(); // Will print the table on the serial port
    Serial.print(",");

    // MQ-3 detecting Benzene
    Serial.print("Benzene:");
    MQ3.update(); // Update data, the arduino will read the voltage from the analog pin
    Serial.print(MQ3.readSensor()); // Sensor will read PPM concentration using the model, a and b values set previously or from the setup
    // MQ3.serialDebug(); // Will print the table on the serial port
    Serial.print(",");

    // HCHO TODO commenting out as giving infs 
    // Serial.print(readHCHO());
    // Serial.print(",");


    Serial.print("Temperature:");
    // BME680 detecting temperature, altitude, humidity, pressure, overall gas content
    Serial.print(bme.temperature); Serial.print(",");

    Serial.print("Pressure:");
    Serial.print(bme.pressure / 100.0); 
    Serial.print(",");

    Serial.print("Humidity:");
    Serial.print(bme.humidity);
    Serial.print(",");

    Serial.print("BME Gas Resistance (VOC):");
    Serial.print(bme.gas_resistance / 1000.0);
    Serial.print(",");

    Serial.print("Altitude:");
    Serial.println(bme.readAltitude(SEALEVELPRESSURE_HPA));

    // wait until next sample ( x1000 for milisecond delay)
    delay(1000/SAMPLING_FREQ_HZ);
    }
    else
      delay(10);
}

double readHCHO()
{
    int sensorValue=analogRead(hchopin);
    double Rs=(4095.0/sensorValue)-1;

    double ppm=pow(10.0,((log10(Rs/HCHOR0)-0.0827)/(-0.4807)));
    return ppm;
}

 
