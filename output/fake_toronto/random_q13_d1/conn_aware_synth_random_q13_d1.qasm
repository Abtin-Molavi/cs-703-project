OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(4.6044849) q[13];
rz(1.8380489) q[11];
rz(2.569949) q[10];
rz(0.32482842) q[6];
rz(3.1823474) q[7];
rz(5.1831813) q[9];
rz(0.95667795) q[4];
cx q[8],q[5];
cx q[1],q[0];
cx q[2],q[3];
