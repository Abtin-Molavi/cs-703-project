OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(0.93760646) q[22];
rz(3.2338022) q[24];
rz(1.9643064) q[25];
rz(4.2768854) q[25];
rz(3.3277433) q[25];
rz(5.4488812) q[25];
cx q[22],q[25];
rz(3.0459784) q[25];
cx q[25],q[22];
rz(0.7231631) q[25];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
rz(4.3047704) q[26];
cx q[26],q[25];
cx q[25],q[26];
cx q[25],q[24];
rz(0.38625588) q[24];
rz(5.5912593) q[24];
rz(5.3987751) q[24];
rz(0.93882221) q[24];
rz(2.8384103) q[25];
rz(1.3582346) q[25];
rz(2.5619911) q[25];
rz(2.0644679) q[25];
rz(4.0562173) q[25];
