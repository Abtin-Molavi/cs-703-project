OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
rz(4.3635364) q[0];
rz(3.9187663) q[1];
cx q[4],q[5];
rz(3.2652282) q[6];
cx q[3],q[7];
rz(0.37469044) q[2];
cx q[2],q[1];
rz(3.6400665) q[7];
rz(5.4783867) q[4];
cx q[5],q[3];
cx q[0],q[6];
cx q[3],q[4];
rz(5.4010681) q[6];
rz(2.4325962) q[5];
cx q[0],q[2];
rz(4.7390299) q[7];
rz(3.5094639) q[1];
cx q[5],q[0];
rz(2.9021127) q[3];
rz(4.9956419) q[6];
cx q[1],q[2];
rz(2.4180241) q[4];
rz(3.4872122) q[7];
cx q[2],q[0];
cx q[7],q[5];
rz(0.30135187) q[6];
cx q[4],q[3];
rz(3.4296625) q[1];
rz(5.7022225) q[7];
rz(6.0006014) q[5];
rz(1.6273586) q[3];
cx q[6],q[1];
rz(0.61569467) q[0];
cx q[4],q[2];
cx q[3],q[5];
rz(3.2409331) q[4];
cx q[7],q[1];
rz(2.0379258) q[2];
cx q[6],q[0];
rz(6.1028146) q[5];
rz(4.1177404) q[2];
cx q[6],q[7];
rz(4.8019252) q[4];
rz(2.5580909) q[0];
cx q[3],q[1];
cx q[5],q[7];
cx q[4],q[6];
rz(0.258807) q[1];
rz(4.2189222) q[3];
rz(2.9035027) q[2];
rz(5.7048098) q[0];
rz(1.8464974) q[5];
rz(0.55136109) q[1];
rz(4.0524319) q[2];
rz(5.7651577) q[7];
cx q[4],q[3];
rz(3.5379412) q[6];
rz(6.2042257) q[0];
