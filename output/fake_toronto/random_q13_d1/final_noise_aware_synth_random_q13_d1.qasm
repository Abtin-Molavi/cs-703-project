OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(4.6044849) q[0];
rz(1.8380489) q[19];
rz(2.569949) q[17];
rz(0.32482842) q[12];
rz(3.1823474) q[13];
rz(5.1831813) q[23];
rz(0.95667795) q[20];
cx q[25],q[24];
cx q[11],q[8];
cx q[7],q[6];
