OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
rz(3.6760706) q[2];
cx q[6],q[5];
rz(2.0945465) q[1];
cx q[0],q[3];
rz(0.84110463) q[7];
rz(5.2279446) q[4];
rz(6.0677614) q[6];
rz(0.35497347) q[5];
rz(3.0397725) q[3];
cx q[2],q[4];
cx q[0],q[7];
rz(4.5256887) q[1];
cx q[5],q[6];
cx q[3],q[7];
cx q[4],q[2];
rz(3.4537672) q[0];
rz(3.0301423) q[1];
cx q[7],q[0];
rz(2.8729497) q[6];
rz(0.55162862) q[3];
rz(4.6844315) q[2];
rz(3.6251281) q[5];
cx q[4],q[1];
cx q[0],q[6];
cx q[2],q[7];
cx q[1],q[4];
rz(0.16956245) q[3];
rz(5.0259899) q[5];
cx q[4],q[2];
cx q[1],q[7];
cx q[6],q[5];
rz(2.1343832) q[0];
rz(1.4594247) q[3];
cx q[7],q[3];
cx q[1],q[4];
rz(5.720841) q[2];
rz(2.8058483) q[5];
cx q[6],q[0];
cx q[1],q[3];
cx q[4],q[0];
cx q[6],q[5];
cx q[2],q[7];