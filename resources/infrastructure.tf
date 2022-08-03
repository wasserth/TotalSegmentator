########### Variables ###########

variable "my_region" {
  type    = string
  default = "us-east-1"
}

variable "avail_zone" {
  type    = string
  default = "us-east-1a"
}

# variable "my_ip" {
#   type    = string
#   default = "161.69.22.122/32"
# }

variable "my_cidr_block" {
  type    = string
  default = "10.0.0.0/24"
}

variable "my_key_pair_name" {
  type    = string
  # default = "terraform-ec2"
  default = "terraform-ec2-east"
  description = "The name of the SSH key to install onto the instances."
}

variable "ssh-key-dir" {
  default     = "~/.ssh/"
  description = "The path to SSH keys - include ending '/'"
}

variable "instance_type" {
  type    = string
#   default = "p2.xlarge"  # only working with "ami-0e3c68b57d50caf64" 
  default = "g4dn.xlarge"  # also works with "ami-057396a15eb04af10"
}

variable "spot_price" {
  type    = string
  default = "0.50"
  description = "The maximum hourly price (bid) you are willing to pay for the specified instance, i.e. 0.10. This price should not be below AWS' minimum spot price for the instance based on the region."
}

# Roughly 0.1 dollar per GB per month
variable "ebs_volume_size" {
  type    = string
  default = "1"
  description = "The Amazon EBS volume size (1 GB - 16 TB)."
}

variable "ami_id" {
  type    = string
  # default = "ami-4c5c6e29" # Default AWS Deep Learning AMI (Ubuntu)
  # default = "ami-030544fb939a57d47" # Default AWS Deep Learning AMI (Ubuntu)  # for eu-central-1
#   default = "ami-0e3c68b57d50caf64" # Default AWS Deep Learning AMI (Ubuntu)  # for us-east-1
  # NVIDIA GPU-Optimized AMI
  # https://aws.amazon.com/marketplace/pp/prodview-7ikjtg3um26wq?sr=0-9&ref_=beagle&applicationId=AWSMPContessa
  default = "ami-057396a15eb04af10"
  description = "The AMI ID to use for each instance. The AMI ID will be different depending on the region, even though the name is the same."
}


########### Main ###########

provider "aws" {
    region = "${var.my_region}"
}

locals {
  avail_zone = "${var.avail_zone}"
}

resource "aws_vpc" "main_vpc" {
    cidr_block = "${var.my_cidr_block}"
    instance_tenancy = "default"
    enable_dns_hostnames = true

    tags = {
        Name = "main_vpc"
    }
}

resource "aws_internet_gateway" "main_vpc_igw" {
    vpc_id = "${aws_vpc.main_vpc.id}"

    tags = {
        Name = "main_vpc_igw"
    }
}

resource "aws_default_route_table" "main_vpc_default_route_table" {
    default_route_table_id = "${aws_vpc.main_vpc.default_route_table_id}"

    route {
        cidr_block = "0.0.0.0/0"
        gateway_id = "${aws_internet_gateway.main_vpc_igw.id}"
    }

    tags = {
        Name = "main_vpc_default_route_table"
    }
}

resource "aws_subnet" "main_vpc_subnet" {
    vpc_id = "${aws_vpc.main_vpc.id}"
    cidr_block = "${var.my_cidr_block}"
    map_public_ip_on_launch = true
    availability_zone  = "${local.avail_zone}"
    tags = {
        Name = "main_vpc_subnet"
    }
}

resource "aws_default_network_acl" "main_vpc_nacl" {
    default_network_acl_id = "${aws_vpc.main_vpc.default_network_acl_id}"
    subnet_ids = ["${aws_subnet.main_vpc_subnet.id}"]

    ingress {
        protocol   = -1
        rule_no    = 1
        action     = "allow"
//        cidr_block = "${var.my_ip}"
        cidr_block = "0.0.0.0/0"
        from_port  = 0
        to_port    = 0
    }

    egress {
        protocol   = -1
        rule_no    = 2
        action     = "allow"
        cidr_block = "0.0.0.0/0"
        from_port  = 0
        to_port    = 0
    }

    tags = {
        Name = "main_vpc_nacl"
    }
}

resource "aws_default_security_group" "main_vpc_security_group" {
    vpc_id = "${aws_vpc.main_vpc.id}"

    # SSH access from anywhere
    ingress {
      from_port = 22
      to_port = 22
      protocol = "tcp"
      cidr_blocks = [
        "0.0.0.0/0"]
    }

    ingress {
      from_port = 80
      to_port = 80
      protocol = "tcp"
      cidr_blocks = [
        "0.0.0.0/0"]
    }

    # for git clone
    egress {
      from_port = 0
      to_port = 0
      protocol = "-1"
      cidr_blocks = [
        "0.0.0.0/0"]
    }

    tags = {
        Name = "main_vpc_security_group"
    }
}

# Use this if want to get a spot instance
#
# resource "aws_spot_instance_request" "aws_dl_custom_spot" {
#     ami                         = "${var.ami_id}"
#     spot_price                  = "${var.spot_price}"
#     instance_type               = "${var.instance_type}"
#     key_name                    = "${var.my_key_pair_name}"
#     monitoring                  = true
#     associate_public_ip_address = true
#     instance_interruption_behavior = "stop"
#     count                       = "1"
#     security_groups             =["${aws_default_security_group.main_vpc_security_group.id}"]
#     subnet_id                   = "${aws_subnet.main_vpc_subnet.id}"
#     ebs_block_device            {
#                                     device_name = "/dev/sdh"
#                                     volume_size = "${var.ebs_volume_size}"
#                                     volume_type = "gp2"
#                                 }

#     # root_block_device {
#     #     volume_size = "${var.root_volume_size}"
#     #     volume_type = "gp2"
#     #     delete_on_termination = true
#     # }

#     tags = {
#         Name = "aws_dl_custom_spot"
#     }
# }

resource "aws_instance" "aws_dl_custom_spot" {
    ami                         = "${var.ami_id}"
    instance_type               = "${var.instance_type}"
    key_name                    = "${var.my_key_pair_name}"
    monitoring                  = true
    associate_public_ip_address = true
    count                       = "1"
    vpc_security_group_ids      = ["${aws_default_security_group.main_vpc_security_group.id}"]
    subnet_id                   = "${aws_subnet.main_vpc_subnet.id}"

    # root_block_device {
    #     volume_size           = 50
    #     delete_on_termination = true
    # }

    ebs_block_device            {
                                device_name = "/dev/sdh"
                                volume_size = "${var.ebs_volume_size}"
                                volume_type = "gp2"
                                }

    tags = {
        Name = "aws_dl_custom_spot"
    }
}



########### Outputs spot instance ###########

# output "id" {
#   value = ["${aws_spot_instance_request.aws_dl_custom_spot.*.id}"]
# }

# output "key-name" {
#   value = ["${aws_spot_instance_request.aws_dl_custom_spot.*.key_name}"]
# }

# output "spot_bid_status" {
#     description = "The bid status of the AWS EC2 Spot Instance request(s)."
#     value       = ["${aws_spot_instance_request.aws_dl_custom_spot.*.spot_bid_status}"]
# }

# output "spot_request_state" {
#     description = "The state of the AWS EC2 Spot Instance request(s)."
#     value       = ["${aws_spot_instance_request.aws_dl_custom_spot.*.spot_request_state}"]
# }

# output "instance-private-ip" {
#   value = ["${aws_spot_instance_request.aws_dl_custom_spot.*.private_ip}"]
# }

# output "instance-public-ip" {
#   value = ["${aws_spot_instance_request.aws_dl_custom_spot.*.public_ip}"]
# }


########### Outputs normal instance ###########

output "id" {
  value = ["${aws_instance.aws_dl_custom_spot.*.id}"]
}

output "key-name" {
  value = ["${aws_instance.aws_dl_custom_spot.*.key_name}"]
}

output "instance-private-ip" {
  value = ["${aws_instance.aws_dl_custom_spot.*.private_ip}"]
}

output "instance-public-ip" {
  value = ["${aws_instance.aws_dl_custom_spot.*.public_ip}"]
}