OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
cx q[1],q[4];
rz(5.4879046) q[6];
rz(2.6366778) q[6];
rz(5.7695558) q[6];
rz(0.92881493) q[6];
cx q[4],q[7];
cx q[4],q[7];
cx q[7],q[4];
cx q[4],q[7];
cx q[4],q[1];
rz(4.6268932) q[10];
cx q[10],q[7];
cx q[7],q[4];
rz(4.0551888) q[7];
rz(6.1968402) q[7];
rz(6.1483685) q[7];
cx q[7],q[10];
cx q[10],q[7];
cx q[7],q[10];
cx q[4],q[7];
cx q[7],q[4];
cx q[4],q[7];
cx q[1],q[4];
rz(1.1927847) q[1];
rz(1.1893294) q[1];
rz(4.2681583) q[1];
rz(1.2796728) q[1];
rz(2.637935) q[1];
rz(5.8345555) q[4];
rz(1.4050016) q[4];
rz(1.276514) q[4];
rz(0.16341934) q[4];
cx q[7],q[4];
cx q[7],q[6];
rz(4.1946529) q[6];
rz(2.8755096) q[6];
rz(5.5307197) q[6];
rz(4.4542032) q[7];
cx q[7],q[4];
cx q[4],q[7];
cx q[7],q[4];
cx q[7],q[10];
rz(2.8414334) q[10];
rz(5.8485624) q[10];
rz(2.9898704) q[10];
cx q[7],q[6];
rz(1.812842) q[6];