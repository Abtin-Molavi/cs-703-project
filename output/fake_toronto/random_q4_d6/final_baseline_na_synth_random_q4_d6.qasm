OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(3.8106785) q[22];
rz(4.3579853) q[22];
rz(2.4614937) q[22];
rz(1.9837864) q[24];
rz(5.6538764) q[24];
rz(2.7384514) q[24];
rz(2.177915) q[24];
rz(5.9724164) q[25];
rz(0.54292076) q[25];
rz(3.3698059) q[25];
rz(1.4456583) q[26];
rz(3.4009637) q[26];
cx q[25],q[26];
cx q[26],q[25];
cx q[25],q[26];
cx q[22],q[25];
cx q[24],q[25];
cx q[25],q[22];
rz(2.3886407) q[22];
cx q[26],q[25];
cx q[25],q[26];
cx q[26],q[25];
cx q[24],q[25];
rz(1.683972) q[25];
cx q[26],q[25];
