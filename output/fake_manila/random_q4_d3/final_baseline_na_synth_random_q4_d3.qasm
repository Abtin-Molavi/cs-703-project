OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
rz(3.1957874) q[0];
rz(2.116506) q[1];
rz(3.2617207) q[1];
rz(2.252118) q[1];
rz(4.9234194) q[4];
rz(4.7045881) q[4];
cx q[3],q[4];
