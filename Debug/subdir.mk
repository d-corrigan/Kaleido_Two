################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../Kaleido.cpp 

OBJS += \
./Kaleido.o 

CPP_DEPS += \
./Kaleido.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/Users/d-corrigan/Downloads/opencv-master/include/opencv -I/usr/local/include/opencv2 -I/usr/local/include/opencv -I/Users/d-corrigan/Downloads/opencv-master/include -I/Users/d-corrigan/Downloads/opencv-master/include/opencv2 -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


