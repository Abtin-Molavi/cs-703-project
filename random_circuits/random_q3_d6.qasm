OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
cx q[2],q[0];
rz(1.8587931) q[1];
rz(1.4873453) q[2];
rz(5.4624288) q[1];
rz(4.6666524) q[0];
rz(5.2356524) q[1];
cx q[2],q[0];
cx q[2],q[1];
rz(2.1132731) q[0];
cx q[0],q[2];
rz(2.3747575) q[1];
rz(1.2729632) q[0];
cx q[2],q[1];
