OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
rz(1.4873453) q[2];
rz(1.8587931) q[3];
rz(5.4624288) q[3];
rz(5.2356524) q[3];
cx q[3],q[2];
rz(2.3747575) q[2];
rz(2.1132731) q[4];
rz(1.2729632) q[4];
cx q[4],q[3];
cx q[3],q[2];
rz(4.6666524) q[2];