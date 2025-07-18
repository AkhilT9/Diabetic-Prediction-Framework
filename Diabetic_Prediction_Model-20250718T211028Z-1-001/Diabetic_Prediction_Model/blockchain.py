import hashlib
import json
from datetime import datetime

class Block:
    def __init__(self, index, timestamp, data, previous_hash):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash() # This will now call the fixed calculate_hash

    def calculate_hash(self):
        # Data to be hashed should *not* include the hash itself,
        # as the hash is derived from the other data.
        # Create a dictionary of the block's core attributes for hashing
        data_to_hash = {
            'index': self.index,
            'timestamp': self.timestamp.isoformat(), # Use ISO format for consistency
            'data': self.data,
            'previous_hash': self.previous_hash
        }
        # Convert the dictionary to a JSON string, sort keys for consistent hashing
        block_string = json.dumps(data_to_hash, sort_keys=True).encode('utf-8')
        return hashlib.sha256(block_string).hexdigest()

    def to_dict(self):
        # Helper to convert block data to a dictionary for JSON serialization
        # Convert datetime to ISO format string for robust serialization/deserialization
        return {
            'index': self.index,
            'timestamp': self.timestamp.isoformat(), # Use ISO format for datetime
            'data': self.data,
            'previous_hash': self.previous_hash,
            'hash': self.hash # Include hash for easier loading, though it's recalculated for validation
        }

class Blockchain:
    def __init__(self):
        self.chain = []
        # Genesis block is created implicitly if no file is loaded, or explicitly if needed
        # It's usually better to create it only if load_chain_from_file fails or if chain is empty
        # after load.

    def create_genesis_block(self):
        """Creates the first block in the chain."""
        if not self.chain: # Only create if chain is empty
            genesis_data = "Genesis Block - Diabetes Prediction History"
            genesis_block = Block(0, datetime.now(), genesis_data, "0")
            self.chain.append(genesis_block)
            print("Genesis Block created.")

    def add_block(self, block):
        """Adds a new block to the chain after validation."""
        if len(self.chain) > 0:
            last_block_hash = self.chain[-1].hash
            if block.previous_hash != last_block_hash:
                print(f"Error: Invalid previous hash for new block! Expected {last_block_hash}, got {block.previous_hash}")
                return False
            # Re-calculate hash to ensure integrity upon addition
            if block.hash != block.calculate_hash():
                print("Error: Block hash mismatch!")
                return False
        self.chain.append(block)
        return True

    def get_last_block(self):
        """Returns the last block in the chain."""
        if not self.chain:
            self.create_genesis_block() # Ensure genesis block exists if chain is empty
        return self.chain[-1]

    def is_chain_valid(self):
        """Verifies the integrity of the entire chain."""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]

            # Check if the current block's hash is correct (recalculate and compare)
            if current_block.hash != current_block.calculate_hash():
                print(f"Chain invalid: Block {current_block.index} hash mismatch.")
                return False
            # Check if the current block points to the correct previous hash
            if current_block.previous_hash != previous_block.hash:
                print(f"Chain invalid: Block {current_block.index} previous hash mismatch.")
                return False
        return True

    def to_list(self):
        """Converts the blockchain to a list of dictionaries for easy display."""
        return [block.to_dict() for block in self.chain]

    def save_chain_to_file(self, filename):
        """Saves the current blockchain to a JSON file."""
        try:
            with open(filename, 'w') as f:
                chain_data = [block.to_dict() for block in self.chain]
                json.dump(chain_data, f, indent=4)
            # print(f"Blockchain saved to {filename}") # Uncomment for verbose saving
        except Exception as e:
            print(f"Error saving blockchain to file {filename}: {e}")

    def load_chain_from_file(self, filename):
        """Loads a blockchain from a JSON file, rebuilding the Block objects."""
        try:
            with open(filename, 'r') as f:
                chain_data = json.load(f)

            new_chain = []
            for block_data in chain_data:
                # Ensure timestamp is converted back to datetime object from ISO string
                block_data['timestamp'] = datetime.fromisoformat(block_data['timestamp'])

                # Reconstruct Block object. The Block.__init__ will recalculate 'hash'
                # to ensure integrity, so we can ignore the 'hash' key from block_data
                # when passing to the constructor.
                loaded_hash = block_data.pop('hash', None)

                block = Block(**block_data)

                # OPTIONAL: Validate loaded_hash vs recalculated hash for extra security during load
                if loaded_hash and loaded_hash != block.hash:
                    print(f"Integrity warning: Block {block.index} hash mismatch during load.")

                new_chain.append(block)

            self.chain = new_chain
            print(f"Blockchain loaded from {filename}. Total blocks: {len(self.chain)}")
            if not self.is_chain_valid():
                print("Warning: Loaded blockchain is not valid! Potential tampering or corruption.")
                return False # Indicate loading failed due to invalidity
            return True # Indicate successful load
        except FileNotFoundError:
            print(f"Blockchain file '{filename}' not found. Will start a new chain.")
            return False # Indicate file not found, app should create new chain
        except json.JSONDecodeError:
            print(f"Error: Blockchain file '{filename}' is corrupted (JSONDecodeError). Starting a new chain.")
            return False
        except Exception as e:
            print(f"An unexpected error occurred loading blockchain from '{filename}': {e}. Starting a new chain.")
            return False
