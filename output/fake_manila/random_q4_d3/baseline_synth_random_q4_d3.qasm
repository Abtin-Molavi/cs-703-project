OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
rz(4.9234194) q[1];
rz(4.7045881) q[1];
rz(3.1957874) q[0];
rz(2.116506) q[3];
rz(3.2617207) q[3];
rz(2.252118) q[3];
cx q[2],q[1];
