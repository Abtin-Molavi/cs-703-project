OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
rz(2.107828) q[4];
rz(2.0055473) q[2];
rz(1.7077848) q[0];
rz(4.4960619) q[6];
cx q[5],q[1];
rz(2.6990391) q[3];
rz(0.029173532) q[5];
rz(0.25965335) q[4];
cx q[2],q[1];
rz(5.5265323) q[6];
cx q[3],q[0];
rz(3.3656412) q[5];
rz(1.204867) q[1];
rz(1.956787) q[3];
cx q[0],q[4];
rz(1.1913562) q[2];
rz(0.68625118) q[6];
rz(4.9243984) q[5];
rz(4.8708128) q[1];
rz(1.1804762) q[3];
cx q[0],q[4];
rz(3.8827036) q[6];
rz(2.4924469) q[2];
cx q[2],q[3];
cx q[4],q[0];
rz(2.8867208) q[5];
cx q[6],q[1];
rz(0.25294708) q[3];
rz(5.3631818) q[2];
rz(4.4116102) q[6];
rz(3.1455066) q[1];
rz(3.5768756) q[4];
cx q[5],q[0];
rz(6.1450556) q[6];
rz(2.0746421) q[1];
rz(0.96853619) q[2];
cx q[4],q[5];
rz(2.3865933) q[0];
rz(4.3272821) q[3];
cx q[1],q[5];
cx q[0],q[4];
rz(2.4881922) q[6];
rz(3.1437346) q[3];
rz(0.67652963) q[2];
cx q[0],q[3];
rz(2.1436138) q[4];
cx q[1],q[6];
cx q[2],q[5];
cx q[2],q[3];
rz(5.554346) q[5];
cx q[0],q[6];
rz(5.9961725) q[4];
rz(5.4766229) q[1];