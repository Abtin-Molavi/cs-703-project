OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
rz(4.6268932) q[2];
rz(5.4879046) q[4];
rz(2.6366778) q[4];
rz(5.7695558) q[4];
rz(0.92881493) q[4];
cx q[3],q[0];
cx q[0],q[1];
cx q[1],q[3];
cx q[2],q[0];
cx q[0],q[1];
cx q[3],q[2];
rz(4.0551888) q[0];
rz(5.8345555) q[2];
rz(1.4050016) q[2];
rz(1.276514) q[2];
rz(0.16341934) q[2];
cx q[1],q[2];
rz(6.1968402) q[0];
rz(6.1483685) q[0];
cx q[2],q[0];
cx q[1],q[4];
rz(4.1946529) q[4];
rz(2.8755096) q[4];
rz(5.5307197) q[4];
cx q[2],q[4];
rz(4.4542032) q[1];
rz(1.1927847) q[3];
rz(1.1893294) q[3];
rz(4.2681583) q[3];
rz(1.2796728) q[3];
rz(2.637935) q[3];
rz(1.812842) q[4];
rz(2.8414334) q[0];
rz(5.8485624) q[0];
rz(2.9898704) q[0];
