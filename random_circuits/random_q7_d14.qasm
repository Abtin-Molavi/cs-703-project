OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
cx q[1],q[5];
cx q[3],q[6];
cx q[0],q[2];
rz(4.1760104) q[4];
cx q[5],q[1];
rz(5.6347273) q[0];
cx q[3],q[2];
cx q[4],q[6];
cx q[3],q[0];
rz(6.0409798) q[5];
rz(3.6144061) q[6];
rz(3.2643106) q[1];
cx q[2],q[4];
cx q[1],q[6];
cx q[4],q[3];
rz(5.0073342) q[5];
rz(0.0098966612) q[2];
rz(0.047285588) q[0];
cx q[5],q[6];
rz(5.8519738) q[1];
rz(1.6306911) q[2];
cx q[4],q[3];
rz(1.4015632) q[0];
rz(2.1959599) q[5];
cx q[2],q[1];
rz(0.99991235) q[0];
rz(0.10470704) q[6];
rz(1.6174338) q[3];
rz(3.6006239) q[4];
cx q[6],q[1];
rz(2.4399529) q[0];
cx q[4],q[2];
rz(4.7603993) q[5];
rz(5.9731557) q[3];
rz(4.1027383) q[0];
cx q[2],q[3];
rz(3.4088526) q[1];
cx q[4],q[5];
rz(0.66209416) q[6];
cx q[4],q[2];
cx q[1],q[6];
rz(1.6776593) q[3];
rz(0.87925028) q[5];
rz(2.1086615) q[0];
rz(6.2247029) q[5];
cx q[3],q[2];
cx q[1],q[0];
rz(2.8972335) q[4];
rz(4.042318) q[6];
rz(5.1725396) q[3];
cx q[1],q[4];
cx q[5],q[0];
rz(3.9596655) q[2];
rz(4.3038741) q[6];
rz(2.2245673) q[2];
cx q[0],q[1];
rz(4.6135065) q[3];
cx q[6],q[4];
rz(6.1305757) q[5];
rz(0.84344814) q[3];
rz(4.7562251) q[6];
rz(5.572687) q[1];
cx q[4],q[5];
rz(2.7484189) q[2];
rz(1.9744512) q[0];
cx q[2],q[1];
cx q[0],q[4];
cx q[3],q[6];
rz(2.1347238) q[5];
