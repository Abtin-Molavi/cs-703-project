OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
cx q[5],q[0];
rz(1.6115648) q[4];
rz(5.7022573) q[3];
rz(2.2787222) q[2];
rz(1.0291029) q[1];
cx q[0],q[5];
cx q[2],q[1];
rz(4.8546333) q[3];
rz(2.5115083) q[4];
rz(2.5391641) q[4];
cx q[0],q[5];
cx q[2],q[1];
rz(2.6110736) q[3];
cx q[2],q[0];
cx q[3],q[4];
cx q[5],q[1];
cx q[3],q[5];
rz(3.7366359) q[2];
rz(5.8883355) q[0];
cx q[1],q[4];
rz(1.1319877) q[1];
cx q[0],q[2];
cx q[4],q[5];
rz(3.6739639) q[3];
rz(5.8976323) q[0];
rz(3.6316869) q[3];
rz(2.489707) q[1];
cx q[4],q[2];
rz(0.40824383) q[5];
cx q[0],q[4];
rz(5.6242543) q[1];
rz(4.9492058) q[3];
rz(2.433846) q[5];
rz(2.0636634) q[2];
rz(5.4063954) q[2];
cx q[4],q[1];
cx q[5],q[3];
rz(3.5975405) q[0];
rz(4.2184801) q[2];
rz(4.4820039) q[0];
cx q[5],q[3];
cx q[1],q[4];
cx q[4],q[2];
cx q[0],q[5];
cx q[1],q[3];
rz(4.4822876) q[2];
cx q[1],q[5];
rz(1.0970363) q[4];
cx q[3],q[0];
cx q[4],q[3];
rz(4.9727136) q[0];
cx q[5],q[2];
rz(5.3935658) q[1];
rz(3.0476508) q[5];
cx q[4],q[2];
rz(5.9394629) q[1];
rz(0.3351518) q[0];
rz(1.6906037) q[3];