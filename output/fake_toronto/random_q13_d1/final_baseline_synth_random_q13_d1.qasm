OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(0.95667795) q[2];
rz(1.8380489) q[3];
rz(3.1823474) q[5];
rz(4.6044849) q[8];
cx q[12],q[13];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[15],q[12];
cx q[16],q[19];
rz(0.32482842) q[21];
rz(2.569949) q[22];
rz(5.1831813) q[24];
