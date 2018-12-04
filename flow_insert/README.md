Tools for insertions into an ecological database.

### -i(str)	:	Input image folder literal path.
### -d(str)	:	Success output folder literal path.
### -q(str)	:	Quarantine output folder literal path. 
### -c(int)	:	Number of processes to spawn		:		default #CPU count
### -b(int)	:	Buffer size		:		default 1
### -p(int)	:	Database port number		: 		default 3306
### -m(int)	:	Clean up and move image files when done(1) or do not touch files(0)	:	default 0
```console
(flow)	/eco_image-master/>python import.py
			-i /input/ -d /delete/ -q /quarantine/ -c 8 -b 20 -p 3306 -m 1
```

