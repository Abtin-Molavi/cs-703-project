OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
rz(0.17226747) q[6];
rz(0.15107585) q[2];
cx q[7],q[3];
cx q[0],q[1];
cx q[4],q[5];
cx q[7],q[3];
rz(5.3680165) q[6];
rz(3.5302603) q[2];
rz(5.1612808) q[0];
rz(2.4220291) q[1];
cx q[4],q[5];
