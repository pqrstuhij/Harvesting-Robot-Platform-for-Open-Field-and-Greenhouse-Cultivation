#include <iostream>
#include <typeinfo>
#include <cstring>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <stdio.h>

#include "ros/ros.h"
//#include "vision/vision_robot.h"

#define SERVER_IP "192.168.1.207"
#define SERVER_PORT 9000
#define MAX_MESSAGE_SIZE 100

int sock;
struct sockaddr_in server_addr;
char message[MAX_MESSAGE_SIZE];
int message_length;
int state = 1;
int count_number = 1;

void tcp_client_start() {

    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == -1) {
        perror("소켓 생성 실패");
        exit(1); 
    }

    // 서버 주소 설정
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = inet_addr(SERVER_IP);
    server_addr.sin_port = htons(SERVER_PORT);

    // 서버에 연결
    if (connect(sock, (struct sockaddr *)&server_addr, sizeof(server_addr)) == -1) {
        perror("연결 실패");
        exit(1);
    }
}

void tcp_client_send(std::string input){

    std::size_t length = input.size();
    if (length >= MAX_MESSAGE_SIZE) {
        length = MAX_MESSAGE_SIZE - 1;
    }
    std::strncpy(message, input.c_str(), length);

    message_length = strlen(message);
    // 개행 문자 제거
    if (message[message_length - 1] == '\n') {
        message[message_length - 1] = '\0';
        message_length--;
    }

    // 서버로 메시지 전송
    if (send(sock, message, message_length, 0) == -1) {
        perror("메시지 전송 실패");

    }
    
}

void tcp_client_read(std::string input){
    std::cout << "서버로부터 받은 응답: ";

    // 서버로부터 응답 받기
    memset(message, 0, sizeof(message));


    int received = recv(sock, message, sizeof(message), 0);


    if(received > 0){
        std::string str(message);
        std::cout << str << std::endl;


        if (str == "ok"){
            tcp_client_send(input);
        }

        if (str == "finish"){
            tcp_client_send(input);
        }
                
    }

}
    


bool robot_move(vision::vision_robot::Request &req, vision::vision_robot::Response &res){
    
    std::string val_1 = std::to_string(req.x);
    std::string val_2 = std::to_string(req.y);
    std::string val_3 = std::to_string(req.z);
    std::string null= " ";
    
    std::string output = val_1 + "," + val_2 + "," + val_3 + "," + null;
    std::cout<<output<<std::endl;

    tcp_client_read(output);
    sleep(1);

    std::string count = std::to_string(count_number);
    tcp_client_read(count);
    count_number += 1;

    
    sleep(1);
    res.state = 100;

    return true;
}

int main(int argc, char **argv){			
// Node Main Function

tcp_client_start();
sleep(1);

ros::init(argc, argv, "tcp_srv"); 	
ros::NodeHandle nh; 	

ros::ServiceServer server = nh.advertiseService("service", &robot_move);
// Initializes Node Name
// Node handle declaration for communication with ROS system
// Declares subscriber. Create subscriber 'ros_tutorial_sub' usythgbing the 'MsgTutorial'
// message file from the 'ros_tutorials_topic' package. The topic name is
// 'ros_tutorial_msg' and the size of the publisher queue is set to 100.

// tcp_client_read();
ros::spin();


// A function for calling a callback function, waiting for a message to be
// received, and executing a callback function when it is received
return 0;
}
