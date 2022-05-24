import RPi.GPIO as GPIO
import time
import sys
def helper_switch(num):
    GPIO.output(5,num&1)
    GPIO.output(6,(num>>1)&1)
    GPIO.output(13,(num>>2)&1)
    
if __name__=="__main__":
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(5,GPIO.OUT)
    GPIO.setup(6,GPIO.OUT)
    GPIO.setup(13,GPIO.OUT)
    dwell_time=0.01
    step=1
    current_dwell_time=0
    last_switch_num=None
    switch_steps=[]
    dwell_times=[]
    filename=sys.argv[1]
    with open(filename,'r') as f:
        for line in f:
            _,switch_num,_=line.split(',')
            switch_num=int(switch_num)
            if switch_num>8:
                continue
            if last_switch_num is None:
                last_switch_num=switch_num
            elif not switch_num==last_switch_num:
                switch_steps.append(last_switch_num)
                dwell_times.append(current_dwell_time)
            current_dwell_time+=dwell_time
            last_switch_num=switch_num
        switch_steps.append(last_switch_num)
        dwell_times.append(current_dwell_time)
    print(switch_steps,dwell_times)
    counter=0
    start_time=time.time()
    helper_switch(switch_steps[0])
    while True:
        current_time=time.time()
        if current_time-start_time>=dwell_times[counter]:
            counter+=1
            if counter>=len(switch_steps):
                break
            helper_switch(switch_steps[counter])
    GPIO.cleanup()



