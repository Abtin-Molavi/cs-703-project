OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(0.82174149) q[22];
rz(6.2050451) q[22];
cx q[22],q[19];
rz(5.2733316) q[19];
rz(2.4846814) q[24];
rz(0.66411251) q[26];
rz(5.3524741) q[26];
cx q[26],q[25];
cx q[25],q[24];
rz(4.5724818) q[25];
rz(1.3958369) q[25];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[25],q[26];
