OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(0.93760646) q[3];
rz(1.9643064) q[2];
rz(4.2768854) q[2];
rz(4.3047704) q[0];
rz(3.2338022) q[1];
rz(3.3277433) q[2];
rz(5.4488812) q[2];
cx q[3],q[2];
cx q[2],q[3];
rz(3.0459784) q[2];
rz(0.7231631) q[2];
cx q[0],q[1];
cx q[1],q[2];
cx q[1],q[0];
rz(2.8384103) q[1];
rz(1.3582346) q[1];
rz(2.5619911) q[1];
rz(2.0644679) q[1];
rz(4.0562173) q[1];
rz(0.38625588) q[2];
rz(5.5912593) q[2];
rz(5.3987751) q[2];
rz(0.93882221) q[2];
