OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
rz(2.8595797) q[2];
cx q[1],q[0];
cx q[0],q[2];
rz(0.89099345) q[1];
rz(3.3804735) q[1];
cx q[2],q[0];
rz(1.5508632) q[0];
rz(2.460613) q[1];
rz(1.1824171) q[2];
rz(3.521409) q[1];
cx q[2],q[0];
rz(1.2204335) q[0];
rz(5.3981152) q[2];
rz(5.902832) q[1];
cx q[2],q[0];
rz(6.0207575) q[1];
rz(0.70697223) q[2];
rz(4.4606297) q[0];
rz(4.4074422) q[1];
rz(2.2145429) q[1];
rz(5.5678502) q[0];
rz(3.4176586) q[2];