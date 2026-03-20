ADAM_ROOT := ../..

.PHONY: all clean serve

all: adam.js

adam.js: $(ADAM_ROOT)/src/adam.c $(ADAM_ROOT)/src/adam_json.c
	$(MAKE) -C $(ADAM_ROOT) wasm
	cp $(ADAM_ROOT)/adam.js $(ADAM_ROOT)/adam.wasm .

serve: all
	@echo "Open http://localhost:8000 in your browser"
	python3 -m http.server 8000

clean:
	rm -f adam.js adam.wasm
