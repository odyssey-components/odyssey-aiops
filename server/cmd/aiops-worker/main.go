package main

import (
	"context"
	"log"
	"os"

	"odyssey/aiops/internal/workflow"
)

func main() {
	if len(os.Args) != 2 {
		log.Fatal("usage: aiops-worker <task-id>")
	}
	if err := workflow.RunWorker(context.Background(), os.Args[1]); err != nil {
		log.Fatal(err)
	}
}
