OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
rz(2.412011) q[0];
rz(4.5153255) q[0];
rz(6.1815207) q[0];
rz(5.4547209) q[0];
rz(5.8728208) q[0];
rz(2.7303074) q[0];
rz(0.49090649) q[1];
rz(4.7214589) q[3];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[3];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[2];
cx q[1],q[0];
cx q[2],q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[3],q[2];
cx q[3],q[2];
cx q[2],q[3];
cx q[3],q[2];
rz(5.9220304) q[4];
rz(2.445447) q[4];
rz(5.0796944) q[4];
rz(0.4960646) q[4];
rz(0.32705788) q[4];
cx q[3],q[4];
rz(2.9122506) q[3];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[3];
rz(5.8792143) q[4];
rz(4.6340834) q[4];
cx q[3],q[4];
cx q[3],q[2];
cx q[2],q[3];
cx q[3],q[2];
cx q[1],q[2];
cx q[0],q[1];
rz(1.5780224) q[0];
rz(2.9788202) q[1];
cx q[2],q[1];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[3];
cx q[1],q[2];
rz(4.4190864) q[1];
cx q[2],q[3];
rz(3.6647608) q[3];
rz(0.76188099) q[3];
rz(6.2150178) q[3];
rz(1.4642231) q[3];
rz(3.2071874) q[3];
rz(5.6875611) q[3];
rz(0.80399538) q[3];
rz(3.1569366) q[4];
rz(4.3491615) q[4];