OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(2.4846814) q[11];
rz(0.82174149) q[3];
rz(6.2050451) q[3];
rz(0.66411251) q[5];
rz(5.3524741) q[5];
cx q[3],q[2];
cx q[5],q[8];
cx q[8],q[11];
cx q[3],q[5];
rz(5.2733316) q[2];
rz(4.5724818) q[8];
rz(1.3958369) q[8];
