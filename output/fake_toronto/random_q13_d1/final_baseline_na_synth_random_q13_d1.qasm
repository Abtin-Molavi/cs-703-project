OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(4.6044849) q[0];
rz(1.8380489) q[1];
cx q[2],q[3];
rz(2.569949) q[4];
rz(0.32482842) q[5];
rz(3.1823474) q[6];
rz(5.1831813) q[7];
rz(0.95667795) q[9];
cx q[8],q[11];
cx q[24],q[25];