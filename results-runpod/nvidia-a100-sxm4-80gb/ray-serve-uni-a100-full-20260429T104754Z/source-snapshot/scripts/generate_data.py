"""Generate prompts.csv and responses.csv with synthetic text data.

Usage:
    python generate_data.py                           # default: 500 rows, 128-512 words
    python generate_data.py --min-words 20 --max-words 80 --suffix short
    python generate_data.py --min-words 1000 --max-words 2000 --suffix long
"""

import argparse
import csv
import random
import string

TOPICS = [
    "machine learning model deployment",
    "database optimization and indexing",
    "REST API design patterns",
    "cloud infrastructure management",
    "data pipeline architecture",
    "microservices communication",
    "authentication and authorization",
    "CI/CD pipeline configuration",
    "container orchestration with Kubernetes",
    "monitoring and observability",
    "frontend performance optimization",
    "distributed systems consensus",
    "message queue integration",
    "caching strategies and invalidation",
    "load balancing algorithms",
    "network security best practices",
    "event-driven architecture",
    "real-time data processing",
    "search engine implementation",
    "graph database modeling",
]

FILLER_SENTENCES = [
    "The system needs to handle high throughput while maintaining low latency across all components.",
    "We should consider the trade-offs between consistency and availability in this context.",
    "The configuration parameters must be validated before the service starts accepting requests.",
    "Error handling should be implemented at every layer to prevent cascading failures.",
    "The deployment pipeline runs automated tests before promoting artifacts to production.",
    "Resource utilization metrics are collected every thirty seconds and aggregated over five-minute windows.",
    "The retry mechanism uses exponential backoff with jitter to avoid thundering herd problems.",
    "Connection pooling reduces overhead by reusing established connections across multiple requests.",
    "The schema migration tool applies changes incrementally and supports rollback operations.",
    "Log aggregation provides centralized visibility into system behavior across all services.",
    "The feature flag system allows gradual rollouts and instant rollbacks without redeployment.",
    "Rate limiting protects backend services from being overwhelmed by excessive traffic.",
    "The serialization format was chosen for its compact size and fast encoding and decoding.",
    "Health check endpoints return detailed status information about each dependency.",
    "The build system caches intermediate artifacts to reduce compilation time on subsequent runs.",
    "Blue-green deployments minimize downtime by switching traffic between identical environments.",
    "The garbage collector tuning reduced pause times from fifty milliseconds to under ten.",
    "Partitioning the dataset across multiple nodes improves both read and write performance.",
    "The queue consumer processes messages in batches of one hundred for better throughput.",
    "TLS certificates are rotated automatically before expiration using the certificate manager.",
    "The API gateway handles request routing, authentication, and protocol translation.",
    "Canary releases expose new code to a small percentage of traffic before full rollout.",
    "The time-series database compresses historical data to reduce storage costs.",
    "Circuit breakers prevent repeated calls to failing services and allow recovery time.",
    "The object storage service provides eleven nines of durability for uploaded files.",
    "Horizontal scaling adds more instances behind the load balancer to handle increased load.",
    "The webhook delivery system retries failed deliveries with configurable backoff intervals.",
    "Data replication across regions ensures availability even during localized outages.",
    "The dependency injection framework manages component lifecycle and wiring automatically.",
    "Structured logging with correlation identifiers enables end-to-end request tracing.",
    "The event sourcing pattern stores every state change as an immutable event in the log.",
    "Prefetching frequently accessed data into memory reduces disk I/O and response times.",
    "The consensus algorithm requires a majority of nodes to agree before committing changes.",
    "Input sanitization prevents injection attacks by escaping special characters in user data.",
    "The scheduler distributes workloads evenly across available worker nodes.",
    "Compression reduces the size of network payloads by up to seventy percent.",
    "The service mesh handles inter-service communication, load balancing, and observability.",
    "Dead letter queues capture messages that cannot be processed for later investigation.",
    "The content delivery network caches static assets at edge locations worldwide.",
    "Idempotency keys ensure that duplicate requests produce the same result without side effects.",
    "The write-ahead log guarantees durability by persisting changes before applying them.",
    "Automated capacity planning adjusts resource allocation based on predicted demand patterns.",
    "The token bucket algorithm provides smooth rate limiting with configurable burst capacity.",
    "Snapshot isolation allows concurrent reads and writes without blocking or dirty reads.",
    "The service registry maintains an up-to-date list of available instances and their endpoints.",
    "Graceful shutdown drains active connections before terminating the process.",
    "The pub-sub system decouples producers from consumers and supports fan-out messaging.",
    "Query optimization analyzes execution plans and selects the most efficient access path.",
    "The configuration server distributes settings to all instances and supports hot reloading.",
    "Chaos engineering experiments validate system resilience by injecting controlled failures.",
]

NOISE_WORDS = [
    "essentially",
    "basically",
    "furthermore",
    "additionally",
    "moreover",
    "specifically",
    "particularly",
    "generally",
    "typically",
    "usually",
    "consequently",
    "therefore",
    "however",
    "nevertheless",
    "meanwhile",
    "accordingly",
    "subsequently",
    "ultimately",
    "presumably",
    "arguably",
]


def add_noise(text: str) -> str:
    """Add random noise to text: typos, extra spaces, filler words."""
    words = text.split()
    result = []
    for w in words:
        if random.random() < 0.03:
            result.append(random.choice(NOISE_WORDS))
        if random.random() < 0.02 and len(w) > 3:
            i = random.randint(1, len(w) - 2)
            w = w[:i] + w[i + 1] + w[i] + w[i + 2 :]
        if random.random() < 0.01:
            w = w + random.choice(string.ascii_lowercase)
        result.append(w)
    return " ".join(result)


def generate_text(min_words: int = 128, max_words: int = 512) -> str:
    mean = (min_words + max_words) / 2
    stddev = (max_words - min_words) / 4
    target = random.gauss(mean, stddev)
    target = max(min_words, min(max_words, int(target)))

    topic = random.choice(TOPICS)
    parts = [f"Regarding {topic}, there are several important aspects to consider."]

    while len(" ".join(parts).split()) < target:
        sentence = random.choice(FILLER_SENTENCES)
        parts.append(sentence)

    text = " ".join(parts)
    words = text.split()[:target]
    text = " ".join(words)
    if not text.endswith("."):
        text += "."

    return add_noise(text)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic benchmark data")
    parser.add_argument("--min-words", type=int, default=128)
    parser.add_argument("--max-words", type=int, default=512)
    parser.add_argument("--num-rows", type=int, default=500)
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="File suffix, e.g. 'short' → prompts-short.csv",
    )
    args = parser.parse_args()

    tag = f"-{args.suffix}" if args.suffix else ""

    prompts_path = f"test-script/prompts{tag}.csv"
    with open(prompts_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user_msg"])
        for _ in range(args.num_rows):
            writer.writerow([generate_text(args.min_words, args.max_words)])

    responses_path = f"test-script/responses{tag}.csv"
    with open(responses_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["assistant_msg"])
        for _ in range(args.num_rows):
            writer.writerow([generate_text(args.min_words, args.max_words)])

    print(
        f"Generated {prompts_path} and {responses_path} "
        f"with {args.num_rows} rows each ({args.min_words}-{args.max_words} words)."
    )


if __name__ == "__main__":
    main()
